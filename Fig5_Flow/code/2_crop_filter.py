import numpy as np
from numpy.fft import fft2, ifft2
import nrrd
from scipy.ndimage import gaussian_filter1d

import imregister


def shift_crop(img, offsets, crop, lim):
    nt = img.shape[0]
    osize = [crop[1] - crop[0], crop[3] - crop[2]]
    out = np.empty((nt, *osize), np.uint8)
    for it in range(nt):
        if it % 100 == 0:
            print(it)
        shifted = ifft2(imregister.subshift(
            fft2(img[it, :, :]),
            offsets[it, :]))
        shifted = shifted[crop[0]:crop[1], crop[2]:crop[3]]
        out[it, :, :] = np.clip(np.round(shifted.real), 0, 255)
        
        # Black-out any wrapping parts.
        dyf, dxf = np.floor(offsets[it, :]).astype(int)
        dyc, dxc = np.ceil(offsets[it, :]).astype(int)
        out[it, :max(0, lim[0]+dyc-crop[0]), :] = 0
        out[it, min(osize[0], osize[0]+lim[1]+dyf-crop[1]):, :] = 0
        out[it, :, :max(0, lim[2]+dxc-crop[2])] = 0
        out[it, :, min(osize[1], osize[1]+lim[3]+dxf-crop[3]):] = 0
    return out


labels = [
    "Vivo_Control_hCG14h_040824",
    "Vivo_Control_hCG14h_041524",
    "Vivo_Control_hCG14h_041724",
    "Vivo_Padrin_hCG14h_041024",
    "Vivo_Padrin_hCG14h_041524",
    "Vivo_Padrin_hCG14h_041724",
]
# The limits are the reasonable part of the image to take from when shifted
# outside the original frame.
crops_limits = [
    # (crop [y0, y1, x0, x1], limits [y0, y1, x0, x1])
    ([110, 110+292, 69, 69+426], [7, 550, 0, 500]),
    ([10, 10+372, 0, 392], [7, 550, 0, 500]),
    ([28, 28+436, 27, 27+444], [7, 550, 0, 500]),
    ([7, 7+399, 75, 75+319], [7, 550, 0, 500]),
    ([57, 57+340, 53, 53+256], [7, 550, 0, 500]),
    ([101, 101+360, 76, 76+424], [7, 550, 0, 500]),
]
time_filter_sigmas = [2, 2, 1, 5, 5, 2]

for i in range(6):
    la = labels[i]
    print(la)
    offsets = np.load(f"offsets/{la}_offsets.npy")
##    from matplotlib import pyplot as plt
##    plt.plot(offsets[:, 0], "g-")
##    plt.plot(offsets[:, 1], "r-")
##    plt.show()
##    continue
    img, header = nrrd.read(f"nrrd/{la}.nrrd", index_order="C")
    img = img[1:, :, :]  # The first frame is junk.
    
    shifted = shift_crop(img, offsets, crops_limits[i][0], crops_limits[i][1])
    nrrd.write(f"registered/{la}_reg.nrrd", shifted, header, index_order="C")
    print("Temporally filtering...")
    # Smooth along time to reduce speckle noise.
    gaussian_filter1d(shifted, time_filter_sigmas[i], 0, output=shifted)
    nrrd.write(f"time_filtered/{la}_tf.nrrd", shifted, header, index_order="C")
