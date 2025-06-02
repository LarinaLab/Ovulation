import numpy as np
from numpy.fft import fft, fft2, fftshift
import nrrd
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d

import imregister


def register(nrrd_path, crop, inspect=False, transfer_threshold=60.):
    arr, header = nrrd.read(nrrd_path, index_order="C")
    sub = arr[1:, crop[0]:crop[1], crop[2]:crop[3]]
    nt = sub.shape[0]

    # These are the offsets to apply to the subsequent frames to place them
    # relative to the first frame. Coordinates are y then x.
    offsets0 = np.empty((nt, 2), dtype=np.float64)
    offsets0[0, :] = 0
    offsets_dif = np.empty((nt, 2), dtype=np.float64)
    offsets_dif[0, :] = 0

    A0 = fft2(sub[0, :, :])
    B = A0

    for it in range(1, nt):
        if it % 100 == 0:
            print(it)
            if inspect:
                plt.plot(offsets0[:it, 0], "g-")
                plt.plot(offsets0[:it, 1], "r-")
                plt.plot(offsets_dif[:it, 0], "g--")
                plt.plot(offsets_dif[:it, 1], "r--")
                plt.show()

        A = B
        B = fft2(sub[it, :, :])
        
        shift, cc = imregister.subregister(A, B, 9)
        offsets_dif[it, :] = offsets_dif[it - 1, :] + shift
        
        shift, cc = imregister.subregister(A0, B, 4)
        offsets0[it, :] = shift

    print("Filtering...")
    # offsets0 tracks the bulk motion effectively, but it has far too much noise frame-to-frame.
    # offsets_dif is the opposite; it tracks frame-to-frame almost perfectly, but tends to drift.
    # We combine the low frequencies of offsets0 with the high frequencies of offsets_dif.
    offsets0_f1 = median_filter(offsets0, (9, 1))
    offsets0_f2 = gaussian_filter1d(offsets0_f1, transfer_threshold, axis=0)
    offsets_dif_f = offsets_dif - gaussian_filter1d(offsets_dif, transfer_threshold, axis=0)
    offsets_net = offsets0_f2 + offsets_dif_f

    if inspect:
        plt.plot(offsets_net[:, 0], "g-")
        plt.plot(offsets_net[:, 1], "r-")
        plt.show()

    return offsets_net


labels = [
    "Vivo_Control_hCG14h_040824",
    "Vivo_Control_hCG14h_041524",
    "Vivo_Control_hCG14h_041724",
    "Vivo_Padrin_hCG14h_041024",
    "Vivo_Padrin_hCG14h_041524",
    "Vivo_Padrin_hCG14h_041724",
]
crops = [
    # [y0, y1, x0, x1]
    [135, 135+161, 119, 119+381],
    [10, 10+173, 5, 5+426],
    [24, 24+239, 59, 59+420],
    [120, 120+149, 50, 50+318],
    [60, 60+256, 82, 82+193],
    [93, 93+176, 220, 220+208],
]

for i in range(6):
    la = labels[i]
    print(la)
    path_in = f"nrrd/{la}.nrrd"
    offsets = register(path_in, crops[i], inspect=False)
    np.save(f"offsets/{la}_offsets.npy", offsets)
