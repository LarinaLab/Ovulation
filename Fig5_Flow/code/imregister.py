import numpy as np
import numpy.typing as npt
from numpy.fft import ifft2, fftshift


def i_to_shift(i, shape):
    return np.asarray(np.unravel_index(i, shape)) - shape//2


def idft_sub(inft: npt.NDArray[np.complex128],
             center: tuple[float], scale: float,
             width: int = 3) -> npt.NDArray[np.complex128]:
    """Calculate the discrete Fourier transform on a small region surrounding a central point.
    
    :param inft: The input image must be fftshifted so that the DC component lies at [nr//2, nc//2].
    :param center: The fractional center coordinate about which to evaluate the DFT, of the form (i_row, i_column).
    :param scale: The real-domain radius about the center on which to evaluate the DFT.
    :param width: The precision of the square output array in samples. Should be 2/scale - 1
    :return: 
    """
    
    assert inft.ndim == 2, "Isnput must be 2D."
    assert len(center) == 2, "Center must be a 2D coordinate."
    assert width > 0, "Width must be positive."
    
    nr, nc = inft.shape
    xr = (np.arange(width) - width//2)*scale + center[0]
    xc = (np.arange(width) - width//2)*scale + center[1]
    a = 2j * np.pi
    dft_r = np.exp(a/nr * xr[:, None] * (np.arange(nr) - nr//2)[None, :])
    dft_c = np.exp(a/nc * xc[None, :] * (np.arange(nc) - nc//2)[:, None])
    return dft_r @ inft @ dft_c


def subregister(A: npt.NDArray[np.complex128],
                B: npt.NDArray[np.complex128],
                iters: int = 7) -> tuple[npt.NDArray[np.float64], np.complex128]:
    """Subpixel image registration by iteratively refined cross correlation.
    
    Inputs
    :param A: Fourier transform of reference image (not fftshifted).
    :param B: Fourier transform of image to register (not fftshifted).
    :param iters: How many iterations to refine the registration by.
        Each iteration halves the error, so 0 will give single pixel
        registration, 1 will give half pixel registration, etc.
    :return: A tuple, (shift, best_cc), containing the vector which will
        shift B onto A, and the largest cross correlation as a complex value. 
    """

    # A and B must be fftshifted.
    shape = np.asarray(A.shape)
    C = A * B.conj()
    cc = fftshift(ifft2(C))
    i_flat = np.argmax(cc)
    shift = i_to_shift(i_flat, shape).astype(np.float64)

    C = fftshift(C)
    scale = 1
    for _ in range(iters):
        scale /= 2
        cc = idft_sub(C, shift, scale)
        i_flat = np.argmax(cc)
        shift += scale * i_to_shift(i_flat, np.array([3, 3]))
    best_cc = cc.flat[i_flat]
    return shift, best_cc


def cc_error(A: npt.NDArray[np.complex128],
             B: npt.NDArray[np.complex128],
             best_cc: np.complex128) -> np.float64:
    return np.sqrt(abs(1 - best_cc*best_cc / (np.sum(A*A) * np.sum(B*B))))


def subshift(B: npt.NDArray[np.complex128],
             shift: npt.NDArray[np.float64],
             best_cc: np.complex128 | None = None) -> npt.NDArray[np.complex128]:
    """Apply the shift in the Fourier domain.

    If you provide best_cc, the overall phase will also be corrected.
    """

    if best_cc is None:
        phase = 0
    else:
        phase = np.angle(best_cc)
    kr = fftshift(np.arange(B.shape[0]) - B.shape[0]//2)[:, None] / B.shape[0]
    kc = fftshift(np.arange(B.shape[1]) - B.shape[1]//2)[None, :] / B.shape[1]
    return B * np.exp(1j * (phase - 2*np.pi*(shift[0]*kr + shift[1]*kc)))
