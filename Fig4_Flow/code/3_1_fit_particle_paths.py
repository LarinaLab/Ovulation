import pickle
import os

import numpy as np
import numpy.typing as npt
import nrrd
from scipy import ndimage
from PIL import Image

from particle import Particle


dir_out = "tracks"
if not os.path.isdir(dir_out):
    os.mkdir(dir_out)
min_particle_sizes: list[float] = [
    2.,
    
    0., 2., 1.5,
    2., 2., 2.,
    1.5, 2., 2.,

    2., 2., 2.,
]
min_chain_lengths: list[int] = [
    8,
    
    4, 5, 8,
    8, 5, 7,
    7, 8, 5,

    8, 8, 8,
]
file_names = [
    "Ampulla_Vivo_hCG17h_101424_mouse1_1",
    
    "Ovary_Vivo_hCG1h_022223",
    "Ovary_Vivo_hCG1h_030123",
    "Ovary_Vivo_hCG1h_033023",
    "Bursa_Vivo_090222",
    "Bursa_Vivo_hCG12P5h_091923",
    "Bursa_Vivo_hCG13h_090523",
    "Ampulla_Vivo_hCG17h_041924_1",
    "Ampulla_Vivo_hCG17h_041924_2",
    "Ampulla_Vivo_hCG17h_042224",

    "Ovary_Vivo_hCG1h_100424_2",
    "Ovary_Vivo_hCG1h_100424_3",
    "Ovary_Vivo_hCG1h_100424_4",
]

for name_index in [2]: #range(len(file_names)):
    la = file_names[name_index]
    print(la)
    min_particle_size = min_particle_sizes[name_index]
    min_chain_len = min_chain_lengths[name_index]
    
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
    mask = mask[None, y0:y1, x0:x1]
    offset = np.array([y0, x0])[None, :]

    # Choices for the centroid estimation:
    # The mask proper
    # The the probability map
    # The original intensity
    # The time filtered intensity <- Let's go with this one for now.

    img, _ = nrrd.read(f"time_filtered/{la}_tf.nrrd", index_order="C")
    img = img[4:-4, y0:y1, x0:x1]
    ##img, _ = nrrd.read(f"particles/{la}_particles.nrrd", index_order="C")
    ##img = img[4:-4, 4:-4, 4:-4]
    labels, _ = nrrd.read(f"labelmaps/{la}_labels.nrrd", index_order="C")

    print("Locating particles...")
    bounds = ndimage.find_objects(labels)

    print("Processing particles...")
    # Particle labels are stored for each time point they intersect.
    frame_labels = [[] for _ in range(labels.shape[0])]
    # Map labels to particles.
    particles: dict[Particle] = {}
    for ip, b in enumerate(bounds):
        if ip % 100000 == 0:
            print(f"{ip} of {len(bounds)}")
        if b is None:
            continue
        t0 = b[0].start
        t1 = b[0].stop
        length = t1 - t0
        if length < min_chain_len:
            continue
        label = ip + 1
        
        voxels = ndimage.sum_labels(img[b], labels[b], label) / 255
        if voxels < min_particle_size:  # Exclude dim or skinny particles.
            continue
        
        particles[label] = part = Particle(label, t0, t1)
        part.voxels = voxels

        local_offset = offset.copy()
        local_offset.flat[0] += b[1].start
        local_offset.flat[1] += b[2].start
        for t in range(t0, t1):
            com = ndimage.center_of_mass(
                img[t, b[1], b[2]],
                labels[t, b[1], b[2]],
                label)
            part.set_at(t, local_offset + com)
            frame_labels[t].append(label)
        part.fit_spline()

    print(f"Tracked {len(particles)} particles.")

##        from matplotlib import pyplot as plt
##        plt.plot(part.pos[:, 1], "r.")
##        plt.plot(part.pos[:, 0], "g.")
##        plt.plot(part.fit_pos[:, 1], "r-")
##        plt.plot(part.fit_pos[:, 0], "g-")
##        plt.show()

    track_data = {
        "frame_labels": frame_labels,
        "particles": particles
    }

    with open(f"{dir_out}/{la}_track.pkl", "wb") as file:
        pickle.dump(track_data, file)
