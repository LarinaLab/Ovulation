import numpy as np
from numpy.fft import fft, fft2, fftshift
import nrrd
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d

import imregister


def register(nrrd_path, crop, inspect=False):
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
    threshold = 60.
    offsets0_f1 = median_filter(offsets0, (9, 1))
    offsets0_f2 = gaussian_filter1d(offsets0_f1, threshold, axis=0)
    offsets_dif_f = offsets_dif - gaussian_filter1d(offsets_dif, threshold, axis=0)
    offsets_net = offsets0_f2 + offsets_dif_f

    if inspect:
        plt.plot(offsets_net[:, 0], "g-")
        plt.plot(offsets_net[:, 1], "r-")
        plt.show()

    return offsets_net


labels = [
    "Ampulla_Vivo_hCG17h_101424_mouse1_1",
    
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
crops = [
    # [y0, y1, x0, x1]
    [18, 18+49, 84, 84+279],
    
##    [16, 16+274, 161, 161+331],
##    [24, 24+413, 139, 139+319], # [24, 24+430, 65, 65+392], ???
##    [7, 7+296, 65, 65+339],
##    [213, 334, 205, 500],  # Part of the base layer. ???
##    [273, 273+156, 123, 123+325],  # The extremely mobile base. ???
##    [30, 30+50, 141, 141+295],  # The almost-static top.
##    [9, 9+180, 16, 16+259],  # The whole channel.
##    [14, 14+84, 0, 420],  # The top membrane.
##    [80, 80+223, 14, 14+161],  # The bulk of the channel.
##
##    [33, 33+332, 55, 55+344],
##    [48, 48+239, 48, 48+373],
##    [39, 39+294, 96, 96+374],
##    [17, 17+214, 48, 48+408],
]

for i in range(1):
    la = labels[i]
    print(la)
    path_in = f"nrrd/{la}.nrrd"
    offsets = register(path_in, crops[i], inspect=False)
    np.save(f"offsets/{la}_offsets.npy", offsets)
