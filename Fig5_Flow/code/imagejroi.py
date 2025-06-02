import struct
import numpy as np
import numpy.typing as npt


def load_imagej_roi_points(path: str) -> npt.NDArray[np.float32]:
    with open(path, "rb") as f:
        data = f.read()

    n_coords = struct.unpack(">h", data[0x10:0x12])[0]
    header2_offset = struct.unpack(">i", data[0x3C:0x40])[0]
    counters_offset = struct.unpack(
        ">i", data[header2_offset + 0x30: header2_offset + 0x34])[0]

    size = 4*n_coords
    base_x = 0x40 + 4*n_coords
    base_y = base_x + size
    xs = struct.unpack(">" + n_coords*"f", data[base_x : base_x + size])
    ys = struct.unpack(">" + n_coords*"f", data[base_y : base_y + size])
    cs = struct.unpack(
        ">" + n_coords*"i", data[counters_offset : counters_offset + size])
    ts = [(c >> 8) - 1 for c in cs]
    points = np.array([xs, ys, ts], np.float32).T
    points = points[np.argsort(points[:, 2]), :]
    return points
