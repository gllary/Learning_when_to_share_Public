from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class MemoryGraphs:
    num_nodes: np.ndarray
    num_edges: np.ndarray
    edge_indices: np.ndarray
    node_attributes: dict[str, np.ndarray]
    edge_attributes: dict[str, np.ndarray]
    sample_ids: np.ndarray

    def edge_slices(self) -> Iterable[np.ndarray]:
        start = 0
        for count in self.num_edges.tolist():
            stop = start + int(count)
            yield self.edge_indices[start:stop]
            start = stop


def _nx_arrays(
    graphs: list[Any],
    sample_ids: list[str],
    *,
    node_attributes: list[str],
    edge_attributes: list[str],
) -> MemoryGraphs:
    if len(graphs) != len(sample_ids):
        raise ValueError("Graph and sample-ID counts differ")
    num_nodes = np.asarray([graph.number_of_nodes() for graph in graphs], dtype=np.int64)
    num_edges = np.asarray([graph.number_of_edges() for graph in graphs], dtype=np.int64)
    nodes: dict[str, list[np.ndarray]] = {name: [] for name in node_attributes}
    edges: dict[str, list[np.ndarray]] = {name: [] for name in edge_attributes}
    edge_indices: list[np.ndarray] = []
    for graph in graphs:
        mapping = {node: index for index, node in enumerate(graph.nodes)}
        for _, values in graph.nodes(data=True):
            for name in node_attributes:
                nodes[name].append(np.asarray(values[name]))
        for source, target, values in graph.edges(data=True):
            edge_indices.append(np.asarray([mapping[source], mapping[target]], dtype=np.int64))
            for name in edge_attributes:
                edges[name].append(np.asarray(values[name]))
    node_arrays = {name: np.asarray(values) for name, values in nodes.items()}
    edge_arrays = {name: np.asarray(values) for name, values in edges.items()}
    indices = np.asarray(edge_indices, dtype=np.int64).reshape(-1, 2)
    return MemoryGraphs(
        num_nodes=num_nodes,
        num_edges=num_edges,
        edge_indices=indices,
        node_attributes=node_arrays,
        edge_attributes=edge_arrays,
        sample_ids=np.asarray(sample_ids).astype(str),
    )


def _append_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if name not in group:
        group.create_dataset(
            name,
            data=values,
            maxshape=(None,) + values.shape[1:],
            chunks=True,
        )
        return
    dataset = group[name]
    old = dataset.shape[0]
    dataset.resize(old + values.shape[0], axis=0)
    dataset[old:] = values


def initialize_store(path: Path, *, model_name: str) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w")
    handle.attrs["schema_version"] = 1
    handle.attrs["model_name"] = model_name
    handle.create_group("node_attributes")
    handle.create_group("edge_attributes")
    handle.create_dataset("num_nodes", shape=(0,), maxshape=(None,), dtype="i8", chunks=True)
    handle.create_dataset("num_edges", shape=(0,), maxshape=(None,), dtype="i8", chunks=True)
    handle.create_dataset("edge_indices", shape=(0, 2), maxshape=(None, 2), dtype="i8", chunks=True)
    text = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset("sample_ids", shape=(0,), maxshape=(None,), dtype=text, chunks=True)
    return handle


def append_graphs(
    handle: h5py.File,
    graphs: list[Any],
    sample_ids: list[str],
    *,
    node_attributes: list[str],
    edge_attributes: list[str],
) -> None:
    batch = _nx_arrays(
        graphs,
        sample_ids,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )
    _append_dataset(handle, "num_nodes", batch.num_nodes)
    _append_dataset(handle, "num_edges", batch.num_edges)
    _append_dataset(handle, "edge_indices", batch.edge_indices)
    _append_dataset(handle, "sample_ids", batch.sample_ids.astype(object))
    for name, values in batch.node_attributes.items():
        _append_dataset(handle["node_attributes"], name, values)
    for name, values in batch.edge_attributes.items():
        _append_dataset(handle["edge_attributes"], name, values)


def read_ids(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        values = handle["sample_ids"][:]
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values])


def load_graphs(path: Path, positions: np.ndarray) -> MemoryGraphs:
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1:
        raise ValueError("Graph positions must be a one-dimensional array")
    with h5py.File(path, "r") as handle:
        all_nodes = np.asarray(handle["num_nodes"][:], dtype=np.int64)
        all_edges = np.asarray(handle["num_edges"][:], dtype=np.int64)
        if np.any(positions < 0) or np.any(positions >= len(all_nodes)):
            raise IndexError("Graph position outside cache")
        node_starts = np.concatenate([[0], np.cumsum(all_nodes[:-1])])
        edge_starts = np.concatenate([[0], np.cumsum(all_edges[:-1])])
        selected_nodes = all_nodes[positions]
        selected_edges = all_edges[positions]
        node_chunks: dict[str, list[np.ndarray]] = {
            name: [] for name in handle["node_attributes"].keys()
        }
        edge_chunks: dict[str, list[np.ndarray]] = {
            name: [] for name in handle["edge_attributes"].keys()
        }
        index_chunks: list[np.ndarray] = []
        for position in positions.tolist():
            node_slice = slice(node_starts[position], node_starts[position] + all_nodes[position])
            edge_slice = slice(edge_starts[position], edge_starts[position] + all_edges[position])
            for name in node_chunks:
                node_chunks[name].append(np.asarray(handle["node_attributes"][name][node_slice]))
            for name in edge_chunks:
                edge_chunks[name].append(np.asarray(handle["edge_attributes"][name][edge_slice]))
            index_chunks.append(np.asarray(handle["edge_indices"][edge_slice], dtype=np.int64))
        # h5py fancy indexing only accepts monotonically increasing indices.
        # Split positions deliberately retain Matbench's declared order, which
        # is not generally sorted, so read IDs one-by-one just like the graph
        # slices above.
        raw_ids = [handle["sample_ids"][int(position)] for position in positions]
    return MemoryGraphs(
        num_nodes=selected_nodes,
        num_edges=selected_edges,
        edge_indices=np.concatenate(index_chunks, axis=0),
        node_attributes={name: np.concatenate(chunks, axis=0) for name, chunks in node_chunks.items()},
        edge_attributes={name: np.concatenate(chunks, axis=0) for name, chunks in edge_chunks.items()},
        sample_ids=np.asarray(
            [value.decode() if isinstance(value, bytes) else str(value) for value in raw_ids]
        ),
    )


def load_graphs_by_ids(path: Path, sample_ids: np.ndarray) -> MemoryGraphs:
    """Load graphs in requested manifest order, preserving official duplicate JIDs."""
    requested = np.asarray(sample_ids).astype(str)
    stored = read_ids(path)
    lookup = {value: index for index, value in enumerate(stored.tolist())}
    if len(lookup) != len(stored):
        raise ValueError("coGN graph cache contains duplicate material IDs")
    missing = sorted(set(requested.tolist()).difference(lookup))
    if missing:
        raise KeyError(f"Manifest material IDs absent from coGN cache: {missing[:10]}")
    positions = np.asarray([lookup[value] for value in requested.tolist()], dtype=np.int64)
    graphs = load_graphs(path, positions)
    if not np.array_equal(graphs.sample_ids, requested):
        raise ValueError("coGN graph cache failed to preserve requested sample order")
    return graphs
