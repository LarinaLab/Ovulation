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
    "Ampulla_Vivo_hCG17h_101424_mouse1_1",
    
    "Ampulla_Vivo_hCG17h_041924_1",
    "Ampulla_Vivo_hCG17h_041924_2",
    "Ampulla_Vivo_hCG17h_042224",
    "Bursa_Vivo_090222",
    "Bursa_Vivo_hCG13h_090523",
    "Bursa_Vivo_hCG12P5h_091923",
    "Ovary_Vivo_hCG1h_022223",
    "Ovary_Vivo_hCG1h_030123",
    "Ovary_Vivo_hCG1h_033023",

    "Ovary_Vivo_hCG1h_100424_1",
    "Ovary_Vivo_hCG1h_100424_2",
    "Ovary_Vivo_hCG1h_100424_3",
    "Ovary_Vivo_hCG1h_100424_4",
]
# The limits are the reasonable part of the image to take from when shifted
# outside the original frame.
crops_limits = [
    # (crop [y0, y1, x0, x1], limits [y0, y1, x0, x1])
    ([11, 11+353, 18, 18+332], [7, 550, 0, 500]),
    
    ([7, 7+457, 271, 271+216], [7, 550, 0, 500]),
    ([7, 7+459, 180, 180+262], [7, 550, 0, 500]),
    ([7, 7+481, 54, 54+357], [7, 550, 0, 500]),
    ([9, 400, 266, 500], [9, 550, 0, 500]),
    ([8, 8+520, 0, 500], [8, 550, 0, 500]),
    ([8, 8+534, 22, 22+409], [7, 550, 0, 500]),
    ([9, 9+390, 0, 326], [8, 550, 0, 500]),
    ([8, 8+489, 5, 5+491], [8, 550, 0, 500]),
    ([7, 7+460, 19, 19+272], [7, 550, 0, 500]),

    ([58, 58+320, 42, 42+438], [7, 550, 0, 500]),
    ([51, 51+307, 58, 58+428], [7, 550, 0, 500]),
    ([40, 40+354, 79, 79+388], [7, 550, 0, 500]),
    ([30, 30+359, 44, 44+453], [7, 550, 0, 500]),
]

for i in [8]:
    la = labels[i]
    print(la)
    offsets = np.load(f"offsets/{la}_offsets.npy")
    img, header = nrrd.read(f"nrrd/{la}.nrrd", index_order="C")
    img = img[1:, :, :]  # The first frame is junk.
    
    shifted = shift_crop(img, offsets, crops_limits[i][0], crops_limits[i][1])
    nrrd.write(f"registered/{la}_reg.nrrd", shifted, header, index_order="C")
    print("Temporally filtering...")
    # Smooth along time to reduce speckle noise.
    gaussian_filter1d(shifted, 5, 0, output=shifted)
    nrrd.write(f"time_filtered/{la}_tf.nrrd", shifted, header, index_order="C")
