import numpy as np
from scipy.spatial import cKDTree

from astropy.coordinates import SkyCoord
from astropy import units as u

from pyphot import msgs

def crossmatch(X1, X2, max_distance=2):
    """Cross-match the values between X1 and X2

    By default, this uses a KD Tree for speed.

    Parameters
    ----------
    X1 : array_like
        first dataset, shape(N1, D)
    X2 : array_like
        second dataset, shape(N2, D)
    max_distance : float (optional)
        maximum radius of search.  If no point is within the given radius,
        then inf will be returned.

    Returns
    -------
    dist, ind: ndarrays
        The distance and index of the closest point in X2 to each point in X1
        Both arrays are length N1.
        Locations with no match are indicated by
        dist[i] = inf, ind[i] = N2
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)

    N1, D = X1.shape
    N2, D2 = X2.shape

    if D != D2:
        msgs.error('Arrays must have the same second dimension')

    kdt = cKDTree(X2)

    dist, ind = kdt.query(X1, k=1, distance_upper_bound=max_distance)

    return dist, ind


def crossmatch_angular(X1, X2, max_distance=2):
    """Cross-match angular values between X1 and X2

    by default, this uses a KD Tree for speed.  Because the
    KD Tree only handles cartesian distances, the angles
    are projected onto a 3D sphere.

    Parameters
    ----------
    X1 : array_like
        first dataset, shape(N1, 2). X1[:, 0] is the RA, X1[:, 1] is the DEC,
        both measured in degrees
    X2 : array_like
        second dataset, shape(N2, 2). X2[:, 0] is the RA, X2[:, 1] is the DEC,
        both measured in degrees
    max_distance : float (optional)
        maximum radius of search, measured in degrees.
        If no point is within the given radius, then inf will be returned.

    Returns
    -------
    dist, ind: ndarrays
        The angular distance (arcsecond) and index of the closest point in X2 to
        each point in X1.  Both arrays are length N1.
        Locations with no match are indicated by
        dist[i] = inf, ind[i] = N2
    """
    X1 = X1 * (np.pi / 180.)
    X2 = X2 * (np.pi / 180.)
    max_distance = max_distance * (np.pi / 180.)

    # Convert 2D RA/DEC to 3D cartesian coordinates
    Y1 = np.transpose(np.vstack([np.cos(X1[:, 0]) * np.cos(X1[:, 1]),
                                 np.sin(X1[:, 0]) * np.cos(X1[:, 1]),
                                 np.sin(X1[:, 1])]))
    Y2 = np.transpose(np.vstack([np.cos(X2[:, 0]) * np.cos(X2[:, 1]),
                                 np.sin(X2[:, 0]) * np.cos(X2[:, 1]),
                                 np.sin(X2[:, 1])]))

    # law of cosines to compute 3D distance
    max_y = np.sqrt(2 - 2 * np.cos(max_distance))
    dist, ind = crossmatch(Y1, Y2, max_y)

    # convert distances back to angles using the law of tangents
    not_inf = ~np.isinf(dist)
    x = 0.5 * dist[not_inf]
    dist[not_inf] = (180. / np.pi * 2 * np.arctan2(x, np.sqrt(np.maximum(0, 1 - x ** 2))))

    return dist*3600., ind

def crossmatch_astropy(ra1, dec1, ra2, dec2, max_distance=2, nthneighbor=1):
    """Cross-match between (ra1, dec1) and (ra2, dec2)

    Parameters
    ----------
    ra1 : array_like
        ra of the first list of targets in units of degrees
    dec1 : array_like
        dec of the first list of targets in units of degrees
    ra2 : array_like
        ra of the second list of targets in units of degrees
    dec2 : array_like
        dec of the second list of targets in units of degrees
    max_distance : float (optional)
        maximum radius of search, measured in degrees.
        If no point is within the given radius, then inf will be returned.

    Returns
    -------
    dist, ind: ndarrays
        The angular distance (arcsecond) and index of the closest point in X2 to
        each point in X1.  Both arrays are length N1.
        Locations with no match are indicated by
        dist[i] = inf, ind[i] = N2
    """

    c1 = SkyCoord(ra=ra1 * u.degree, dec=dec1 * u.degree)
    c2 = SkyCoord(ra=ra2 * u.degree, dec=dec2 * u.degree)

    idx, d2d, d3d = c1.match_to_catalog_sky(c2)

    return idx[d2d.arcsec<=max_distance*u.degree], idx[d2d.arcsec>max_distance*u.degree]