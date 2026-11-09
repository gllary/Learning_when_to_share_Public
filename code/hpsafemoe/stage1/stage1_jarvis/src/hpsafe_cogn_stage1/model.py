from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any


OFFICIAL_MODEL_CONFIG = {
    "name": "coGN",
    "inputs": {
        "offset": {
            "shape": (None, 3),
            "name": "offset",
            "dtype": "float32",
            "ragged": True,
        },
        "cell_translation": None,
        "affine_matrix": None,
        "voronoi_ridge_area": None,
        "atomic_number": {
            "shape": (None,),
            "name": "atomic_number",
            "dtype": "int32",
            "ragged": True,
        },
        "frac_coords": None,
        "coords": None,
        "multiplicity": {
            "shape": (None,),
            "name": "multiplicity",
            "dtype": "int32",
            "ragged": True,
        },
        "lattice_matrix": None,
        "edge_indices": {
            "shape": (None, 2),
            "name": "edge_indices",
            "dtype": "int32",
            "ragged": True,
        },
        "line_graph_edge_indices": None,
    },
}


def build_model() -> Any:
    logging.getLogger("kgcnn").setLevel(logging.WARNING)
    from kgcnn.literature.coGN import make_model

    # Passing only the same minimal config as the official JARVIS contribution
    # lets kgcnn 3.0.1 fill its frozen coGN defaults through update_model_kwargs.
    return make_model(**deepcopy(OFFICIAL_MODEL_CONFIG))


def build_latent_model(model: Any, *, expected_dim: int = 128) -> Any:
    """Pool the trained final coGN message-passing node state by multiplicity."""
    import tensorflow as tf

    candidates = [
        layer for layer in model.layers if layer.__class__.__name__ == "SequentialGraphNetwork"
    ]
    if not candidates:
        raise RuntimeError("Cannot find coGN SequentialGraphNetwork latent layer")
    output = candidates[-1].output
    if not isinstance(output, (tuple, list)) or not isinstance(output[1], dict):
        raise RuntimeError("Unexpected kgcnn 3.0.1 coGN intermediate output")
    node_features = output[1]["features"]
    multiplicity = tf.cast(output[1]["multiplicity"], node_features.dtype)
    weighted = node_features * tf.expand_dims(multiplicity, axis=-1)
    numerator = tf.reduce_sum(weighted, axis=1)
    denominator = tf.reduce_sum(multiplicity, axis=1, keepdims=True)
    latent = numerator / tf.maximum(denominator, tf.cast(1e-8, numerator.dtype))
    latent_model = tf.keras.Model(model.inputs, latent, name="cogn_multiplicity_latent")
    if int(latent_model.output_shape[-1]) != expected_dim:
        raise RuntimeError(
            f"Official coGN latent width {latent_model.output_shape[-1]} != {expected_dim}"
        )
    return latent_model


def configure_tensorflow() -> None:
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    if len(devices) != 1:
        raise RuntimeError(
            f"Each isolated coGN job must see exactly one GPU; TensorFlow sees {len(devices)}"
        )
    tf.config.experimental.set_memory_growth(devices[0], True)
    tf.keras.mixed_precision.set_global_policy("float32")
