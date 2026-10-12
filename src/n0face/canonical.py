"""MediaPipe's canonical (generic, anonymous) 3D face mesh, read from the bundled model file.

Vertex i matches face landmark i, so the same index constants the effects use (eyes, mouth,
constellation nodes) work on it. Used for the interactive mask preview; it's no one's face.
"""

import struct
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from n0face.detector import DEFAULT_MODEL

METADATA_ENTRY = "geometry_pipeline_metadata_landmarks.binarypb"


@dataclass(frozen=True)
class CanonicalFace:
    vertices: np.ndarray  # (468, 3) float32, centimeters, x right, y up, z toward the viewer
    uv: np.ndarray  # (468, 2)
    triangles: np.ndarray  # (898, 3) int32
    edges: np.ndarray  # (N, 2) unique triangle edges


def _varint(b: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        c = b[i]
        i += 1
        result |= (c & 0x7F) << shift
        shift += 7
        if c < 0x80:
            return result, i


def _fields(b: bytes):
    """Minimal protobuf wire-format reader: yields (field number, wire type, value)."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(b, i)
        elif wire == 1:
            value, i = b[i : i + 8], i + 8
        elif wire == 2:
            n, i = _varint(b, i)
            value, i = b[i : i + n], i + n
        elif wire == 5:
            value, i = b[i : i + 4], i + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield field, wire, value


@lru_cache(maxsize=1)
def load(model_path: Path = DEFAULT_MODEL) -> CanonicalFace:
    # GeometryPipelineMetadata.canonical_mesh (field 1) -> Mesh3d:
    #   vertex_buffer (3): x, y, z, u, v per vertex; index_buffer (4): triangle indices.
    data = zipfile.ZipFile(model_path).read(METADATA_ENTRY)
    mesh = next(v for f, _, v in _fields(data) if f == 1)
    floats: list[float] = []
    indices: list[int] = []
    for field, wire, value in _fields(mesh):
        if field == 3:
            floats += struct.unpack(f"<{len(value) // 4}f", value) if wire == 2 else struct.unpack("<f", value)
        elif field == 4:
            if wire == 2:
                j = 0
                while j < len(value):
                    x, j = _varint(value, j)
                    indices.append(x)
            else:
                indices.append(value)

    buf = np.array(floats, dtype=np.float32).reshape(-1, 5)
    tris = np.array(indices, dtype=np.int32).reshape(-1, 3)
    pairs = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    return CanonicalFace(vertices=buf[:, :3], uv=buf[:, 3:], triangles=tris, edges=np.unique(pairs, axis=0))
