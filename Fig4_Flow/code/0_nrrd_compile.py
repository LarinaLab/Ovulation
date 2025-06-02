import numpy as np
import tifffile
import nrrd


def tiff_compile(header, dir_root, label):
    with open(f"nrrd/{label}.nrrd", "wb") as file:
        nrrd.writer._write_header(file, header)
        for i in range(N_FRAMES):
            if i % 100 == 0:
                print(i)
            d = 1 + i // 5000  # Directory label
            j = i + 1  # First image label
            #path_in = f"{dir_root}/{d}/image_{j:06d}.tiff"
            path_in = f"{dir_root}/image_{j:06d}.tiff"
            img = tifffile.imread(path_in)
            img = img[crop_y[0]:crop_y[1], :]
            file.write(img.tobytes())


N_FRAMES = 5000
crop_y = [2, 552]
dir_roots = [
    "COCs in Ampulla/101424_Vivo_hCG17h_500_2_BscanPeriod8p353_1",
    "COCs in Ampulla/101424_Vivo_hCG17h_500_2_BscanPeriod8p353_2",
    "COCs in Ampulla/101424_Vivo_mouse2_hCG17h_500_2_BscanPeriod8p353_3",
    "COCs in Ampulla/101424_Vivo_mouse2_hCG17h_500_2_BscanPeriod8p353_4",
    
##    "COCs in Ampulla/041924_Vivo_hCG17h_500_2_Bscan8p353",
##    "COCs in Ampulla/041924_Vivo_hCG17h_500_2_Bscan8p353_mouse2",
##    "COCs in Ampulla/042224_Vivo_hCG17h_500_2_Bscan8p353",
##    "COCs in Bursa/090222_Vivo_500_2_Bscan8p353",
##    "COCs in Bursa/090523_Vivo_hCG13h_500_2_Bscan8p353",
##    "COCs in Bursa/091923_Vivo_hCG12P5h_500_2_Bscan8p353",
##    "COCs in Ovary/022223_Vivo_hCG1h_500_2_Bscan8p353",
##    "COCs in Ovary/030123_Vivo_hCG1h_500_2_Bscan8p353",
##    "COCs in Ovary/033023_Vivo_hCG1h_500_2_Bscan8p353",
##    
##    "COCs in Ovary/100424_Vivo_hCG1h_500_2_BscanPeriod8p353_1_tiffs",
##    "COCs in Ovary/100424_Vivo_hCG1h_500_2_BscanPeriod8p353_2_tiffs",
##    "COCs in Ovary/100424_Vivo_hCG1h_500_2_BscanPeriod8p353_3_tiffs",
##    "COCs in Ovary/100424_Vivo_hCG1h_500_2_BscanPeriod8p353_4_tiffs",
]
labels = [
    "Ampulla_Vivo_hCG17h_101424_mouse1_1",
    "Ampulla_Vivo_hCG17h_101424_mouse1_2",
    "Ampulla_Vivo_hCG17h_101424_mouse2_3",
    "Ampulla_Vivo_hCG17h_101424_mouse2_4",
    
##    "Ampulla_Vivo_hCG17h_041924_1",
##    "Ampulla_Vivo_hCG17h_041924_2",
##    "Ampulla_Vivo_hCG17h_042224",
##    "Bursa_Vivo_090222",
##    "Bursa_Vivo_hCG13h_090523",
##    "Bursa_Vivo_hCG12P5h_091923",
##    "Ovary_Vivo_hCG1h_022223",
##    "Ovary_Vivo_hCG1h_030123",
##    "Ovary_Vivo_hCG1h_033023",
##
##    "Ovary_Vivo_hCG1h_100424_1",
##    "Ovary_Vivo_hCG1h_100424_2",
##    "Ovary_Vivo_hCG1h_100424_3",
##    "Ovary_Vivo_hCG1h_100424_4",
]
dates = [
    "2024-10-14",
    "2024-10-14",
    "2024-10-14",
    "2024-10-14",
    
##    "2024-04-19",
##    "2024-04-19",
##    "2024-04-22",
##    "2022-09-02",
##    "2023-09-05",
##    "2023-09-19",
##    "2023-02-22",
##    "2023-03-01",
##    "2023-03-30",
##
##    "2024-10-04",
##    "2024-10-04",
##    "2024-10-04",
##    "2024-10-04",
]
ages = [
    "hCG17h",
    "hCG17h",
    "hCG17h",
    "hCG17h",
    
##    "hCG17h",
##    "hCG17h",
##    "hCG17h",
##    "hCG13h",
##    "hCG13h",
##    "hCG12P5h",
##    "hCG1h",
##    "hCG1h",
##    "hCG1h",
##
##    "hCG1h",
##    "hCG1h",
##    "hCG1h",
##    "hCG1h",
]
COC_locations = [
    "Ampulla",
    "Ampulla",
    "Ampulla",
    "Ampulla",
    
##    "Ampulla",
##    "Ampulla",
##    "Ampulla",
##    "Bursa",
##    "Bursa",
##    "Bursa",
##    "Ovary",
##    "Ovary",
##    "Ovary",
##
##    "Ovary",
##    "Ovary",
##    "Ovary",
##    "Ovary",
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

for i in range(len(labels)):
    header["acquisition date"] = dates[i]
    header["age"] = ages[i]
    header["COCs location"] = COC_locations[i]
    tiff_compile(header, dir_roots[i], labels[i])
