import numpy as np
import tifffile
import nrrd


def tiff_compile(header, dir_root, label):
    with open(f"nrrd/{label}.nrrd", "wb") as file:
        nrrd.writer._write_header(file, header)
        for i in range(N_FRAMES):
            if i % 100 == 0:
                print(i)
            j = i + 1  # First image label
            path_in = f"{dir_root}/image_{j:06d}.tiff"
            img = tifffile.imread(path_in)
            img = img[crop_y[0]:crop_y[1], :]
            file.write(img.tobytes())


N_FRAMES = 5000
crop_y = [2, 552]
dir_roots = [
    "Inhibitor/041024_Vivo_Padrin_hCG14h_500_2_Bscan8p353_2_tiffs",
    "Inhibitor/041524_Vivo_Padrin_hCG14h_500_2_Bscan8p353_tiffs",
    "Inhibitor/041724_Vivo_Padrin_hCG14h_500_2_Bscan8p353_2_tiffs",
    "Vehicle/040824_Vivo_Control_hCG14h_500_2_Bscan8p353_tiffs",
    "Vehicle/041524_Vivo_Control_hCG14h_500_2_Bscan8p353_tiffs",
    "Vehicle/041724_Vivo_Control_hCG14h_500_2_Bscan8p353_2_tiffs",
]
labels = [
    "Vivo_Padrin_hCG14h_041024",
    "Vivo_Padrin_hCG14h_041524",
    "Vivo_Padrin_hCG14h_041724",
    "Vivo_Control_hCG14h_040824",
    "Vivo_Control_hCG14h_041524",
    "Vivo_Control_hCG14h_041724",
]
dates = [
    "2024-04-10",
    "2024-04-15",
    "2024-04-17",
    "2024-04-08",
    "2024-04-15",
    "2024-04-17",
]
treatments = [
    "Padrin",
    "Padrin",
    "Padrin",
    "Control",
    "Control",
    "Control",
]
header = {
    "type": "uint8",
    "dimension": 3,
    "space dimension": 3,
    "sizes": [500, crop_y[1] - crop_y[0], N_FRAMES],
    "encoding": "raw",
    "space dimension": 3,
    "space directions": [(5.08,0,0), (0,2.4,0), (0,0,8.353)],
    "space units": ["microns", "microns", "ms"],
    "kinds": ["space", "space", "time"],
    "labels": ["x-galvo", "depth", "time"],
}

for i in range(6):
    header["acquisition date"] = dates[i]
    header["treatment"] = treatments[i]
    tiff_compile(header, dir_roots[i], labels[i])
