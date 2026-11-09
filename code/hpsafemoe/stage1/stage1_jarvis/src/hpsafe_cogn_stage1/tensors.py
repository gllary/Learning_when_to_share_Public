from __future__ import annotations

from typing import Any

import numpy as np

from hpsafe_cogn_stage1.graph_store import MemoryGraphs


def input_tensors(model: Any, graphs: MemoryGraphs) -> dict[str, Any]:
    """Convert the locked graph cache to the exact kgcnn ragged inputs."""
    import tensorflow as tf
    from kgcnn.graph.methods import get_angle_indices

    input_names = [value.name.split(":", maxsplit=1)[0] for value in model.inputs]
    tensors: dict[str, Any] = {}
    for name, values in graphs.edge_attributes.items():
        if name in input_names:
            tensors[name] = tf.RaggedTensor.from_row_lengths(values, graphs.num_edges)
    for name, values in graphs.node_attributes.items():
        if name in input_names:
            tensors[name] = tf.RaggedTensor.from_row_lengths(values, graphs.num_nodes)
    # Cache edges are NetworkX (source, target); kgcnn GraphDict uses (target, source).
    model_edges = graphs.edge_indices[:, [1, 0]]
    tensors["edge_indices"] = tf.RaggedTensor.from_row_lengths(model_edges, graphs.num_edges)
    if "line_graph_edge_indices" in input_names:
        line_indices = [
            get_angle_indices(edge_indices[:, [1, 0]], edge_pairing="kj")[2].reshape(-1, 2)
            for edge_indices in graphs.edge_slices()
        ]
        nonempty = [value for value in line_indices if len(value)]
        concatenated = (
            np.concatenate(nonempty, axis=0)
            if nonempty
            else np.empty((0, 2), dtype=np.int64)
        )
        tensors["line_graph_edge_indices"] = tf.RaggedTensor.from_row_lengths(
            concatenated, np.asarray([len(value) for value in line_indices], dtype=np.int64)
        )
    missing = sorted(set(input_names) - set(tensors))
    if missing:
        raise ValueError(f"Graph cache cannot supply model inputs: {missing}")
    # CrystalDataset.tensor casts each declared input to its configured dtype.
    for value in model.inputs:
        name=value.name.split(':',maxsplit=1)[0]
        tensors[name]=tf.cast(tensors[name],value.dtype)
    return tensors
