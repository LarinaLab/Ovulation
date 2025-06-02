from math import ceil

import nrrd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


g = torch.Generator()
g.manual_seed(423978)

bs = 9
r = bs // 2

# A particle detector model:
model = nn.Sequential(
    nn.Conv3d(1, 1, kernel_size=3),
    nn.Conv3d(1, 1, kernel_size=3),
    nn.SELU(inplace=True),
    nn.Conv3d(1, 1, kernel_size=3),
    nn.Conv3d(1, 1, kernel_size=3),
    nn.Sigmoid()
)
model.load_state_dict(torch.load(
    "../Fig5_Flow/particle_detector_v0/adaptive_filter/particle_detector.model"))

labels = [
##    "Ampulla_Vivo_hCG17h_101424_mouse1_1",
    
##    "Ampulla_Vivo_hCG17h_041924_1",
##    "Ampulla_Vivo_hCG17h_041924_2",
##    "Ampulla_Vivo_hCG17h_042224",
##    "Bursa_Vivo_090222",
##    "Bursa_Vivo_hCG13h_090523",
##    "Bursa_Vivo_hCG12P5h_091923",
##    "Ovary_Vivo_hCG1h_022223",
    "Ovary_Vivo_hCG1h_030123",
##    "Ovary_Vivo_hCG1h_033023",
##
##    "Ovary_Vivo_hCG1h_100424_2",
##    "Ovary_Vivo_hCG1h_100424_3",
##    "Ovary_Vivo_hCG1h_100424_4",
]

tbs = 1000
step = tbs - 2*r

for label in labels:
    print(label)
    img, nrrd_header = nrrd.read(f"registered/{label}_reg.nrrd", index_order="C")

    mean = img.mean().astype(np.float32)
    std = img.std().astype(np.float32)
    img = torch.tensor((img - mean) / std)

    mask = np.array(Image.open(f"registered/particle_masks/{label}_mask.png"))
    mask = mask > 127
    masked_indices = np.argwhere(mask)
    y0, x0 = masked_indices.min(axis=0)
    y1, x1 = masked_indices.max(axis=0)
    y0 = max(y0 - r, 0)
    x0 = max(x0 - r, 0)
    y1 = min(y1 + r, mask.shape[0])
    x1 = min(x1 + r, mask.shape[1])
    mask = mask[y0:y1, x0:x1]
    img = img[:, y0:y1, x0:x1]

    out = np.zeros(img.shape, np.uint8)
    for i in range(1 + ceil((img.shape[0] - tbs) / step)):
        print(i*step, i*step + tbs)
        y = model(img[None, None, i*step : i*step + tbs, :, :])
        y = y.detach().numpy().squeeze()
        out[r + i*step : r + i*step + y.shape[0], r:-r, r:-r] = 255 * y
    out *= mask
    nrrd.write(f"particles/{label}_particles.nrrd", out,
               header=nrrd_header, index_order="C")
