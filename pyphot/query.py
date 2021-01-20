from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

def twomass(ra, dec, radius='60m'):
    Vizier.ROW_LIMIT = -1
    result = Vizier.query_region(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'), radius=radius,
                                 catalog=["II/246"])
    return result[0]
