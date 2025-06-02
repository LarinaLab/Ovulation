import pickle

import numpy as np
import nrrd
from scipy import ndimage
from PIL import Image


ALPHA = 80
labels = [
##    "Ampulla_Vivo_hCG17h_101424_mouse1_1",
    
##    "Ovary_Vivo_hCG1h_022223",
    "Ovary_Vivo_hCG1h_030123",
##    "Ovary_Vivo_hCG1h_033023",
##    "Bursa_Vivo_090222",
##    "Bursa_Vivo_hCG12P5h_091923",
##    "Bursa_Vivo_hCG13h_090523",
##    "Ampulla_Vivo_hCG17h_041924_1",
##    "Ampulla_Vivo_hCG17h_041924_2",
##    "Ampulla_Vivo_hCG17h_042224",
##
##    "Ovary_Vivo_hCG1h_100424_2",
##    "Ovary_Vivo_hCG1h_100424_3",
##    "Ovary_Vivo_hCG1h_100424_4",
]

for la in labels:
    print(la)

    mask = np.array(Image.open(f"registered/particle_masks/{la}_mask.png"))
    mask = mask > 127
    # Find the bounds of the ROI. The particle invalid region is size 4.
    masked_indices = np.argwhere(mask)
    y0, x0 = masked_indices.min(axis=0)
    y1, x1 = masked_indices.max(axis=0)
    y0 = max(y0 - 4, 0) + 4
    x0 = max(x0 - 4, 0) + 4
    y1 = min(y1 + 4, mask.shape[0]) - 4
    x1 = min(x1 + 4, mask.shape[1]) - 4

    with open(f"tracks/{la}_track.pkl", "rb") as file:
        track_data = pickle.load(file)
    particles = track_data["particles"]
    labels, header = nrrd.read(f"labelmaps/{la}_labels.nrrd", index_order="C")
    out = np.zeros((labels.shape[0], *mask.shape, 4,), np.uint8)

    print("Locating particles...")
    bounds = ndimage.find_objects(labels)

    print("Filling in particles...")
    for ip, b in enumerate(bounds):
        if ip % 100000 == 0:
            print(f"{ip} of {len(bounds)}")
        if b is None:
            continue
        label = ip + 1
        if label not in particles:
            continue

        mask = labels[b] == label
        out[:, y0:y1, x0:x1][b][mask, :3] = particles[label].get_color()
        out[:, y0:y1, x0:x1][b][mask, 3] = ALPHA

    print("Saving color maps...")
    header["kinds"] = ["RGBA-color"] + header["kinds"]
    header["labels"] = ["label"] + header["labels"]
    nrrd.write(f"labelmaps/{la}_track.nrrd", out, header=header, index_order="C")
    with open(f"tracks/{la}_track.pkl", "wb") as file:
        pickle.dump(track_data, file)
