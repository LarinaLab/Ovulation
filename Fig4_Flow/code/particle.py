from matplotlib.colors import hsv_to_rgb
import numpy.typing as npt
import numpy as np
from scipy.spatial.distance import cdist
from tps import ThinPlateSpline


class Particle:
    """What does a particle need to do?

    Tracking:
    * Store positional data
    * Perform spline fitting
    * Differentiate its spline

    Analysis:
    * Generate and remember its color so we can draw it consistently.
    *
    """

    REGULARIZATION = 2000  # Degree of smoothing for particle paths.
    
    def __init__(self, label: np.int32, t0: int, t1: int) -> None:
        self.label: np.int32
        self.t0: int = t0
        # Positions are (y-, x-coordinate).
        # Units are pixels in the registered frame coordinate space.
        self.pos: npt.NDArray[np.float32] = np.empty((t1 - t0, 2), np.float32)
        self.pos_fit: npt.NDArray[np.float32] | None = None
        # Units are pixels per frame.
        self.vel: npt.NDArray[np.float32] | None = None
        self.color: npt.NDArray[np.uint8] | None = None
        self.voxels: float = float("nan")

    def __len__(self) -> int:
        return self.pos.shape[0]

    def get_color(self) -> npt.NDArray[np.uint8]:
        if self.color is None:
            h = np.random.uniform(0, 1)
            s = np.random.uniform(0.3, 1)
            v = np.random.uniform(0.5, 1)
            self.color = (255 * hsv_to_rgb((h, s, v))).astype(np.uint8)
        return self.color

    def t(self) -> npt.NDArray[np.int32]:
        return np.arange(self.t0, self.t0 + len(self))

    def fit_spline(self) -> None:
        t = self.t()
        
        spline = ThinPlateSpline(self.REGULARIZATION)
        spline.fit(t, np.array(self.pos))

        self.fit_pos = spline.transform(t)
        self.vel = dif_tps(spline, t)

    def set_at(self, time: int, pos: npt.NDArray[np.float32]) -> None:
        self.pos[time - self.t0, :] = pos

    def pos_at(self, time: int) -> npt.NDArray[np.float32]:
        return self.fit_pos[time - self.t0, :]

    def vel_at(self, time: int) -> npt.NDArray[np.float32]:
        return self.vel[time - self.t0, :]


def dif_phi(t: npt.NDArray, c: npt.NDArray) -> npt.NDArray:
    """Compute the two spatial derivatives of pairwise radial distance between
    the given points and the control points evaluated at the given points.

    Used for calculating the surface normal of a thin plate spline.

    Definition:
        d/dt phi(r) = d/dt (r^2 * log(r))
        Where "r" is the distance between the control point and the data point.

    :param t: (n, 1) matrix of n points in the source space.
    :param c: (n_c, 1) matrix of n_c control points.

    :return: A matrix with shape (n, n_c) giving the change in radial distance between
        each point and a control point (d/dt phi_c(t)).
    """

    # r2 is the squared distance, i.e., r^2.
    r2 = cdist(t, c, metric="sqeuclidean")
    r2[r2 == 0] = 1  # Avoid ln(0)
    d_phi_dt = (t - c.T) * (np.log(r2) + 1)
    return d_phi_dt


def dif_tps(tps: ThinPlateSpline, t: npt.NDArray) -> npt.NDArray:
    """Calculate the spatial derivatives of a thin plate spline at some points.

    :param tps: The thin plate spline to use. Must map 2->1 dimensions.
    :param t: The (n, 1) input time coordinates to sample at.
    :return: The (n, 2) derivatives, (dx/dt, dy/dt), for the n points.
    """

    if t.ndim == 1:
        t = t[:, None]

    m = tps.parameters[-1, :]  # Slopes, [dx/dt, dy/dt]
    d_phi_dt = dif_phi(t, tps.control_points)
    d = m + d_phi_dt @ tps.parameters[:-2, :]
    return d
