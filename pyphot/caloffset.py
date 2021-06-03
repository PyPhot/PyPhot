import numpy as np
import matplotlib.pyplot as plt

def degtorad(d):
    """
      Convert degrees into radians.

      Parameters
      ----------
      d : float or array
          Angle in degrees.

      Returns
      -------
      Angle : float or array
          The angle converted into radians.
    """
    return (d / 180.0) * np.pi


def radtodeg(r):
    """
      Convert radians into degrees.

      Parameters
      ----------
      d : float or array
          Angle in radians.

      Returns
      -------
      Angle : float or array
          The angle converted into degrees.
    """
    return (r / np.pi) * 180.0


def positionAngle(ra1, dec1, ra2, dec2, plot=False, positive=False):
    """
      Compute the position angle.

      The position angle is measured from the first position
      from North through East. If the `positive` flag is set
      True (default) the result will be given as an angle from
      0 to 360 degrees.

      Parameters
      ----------
      ra1 : float
          Right ascension of first object [deg].
      dec1 : float
          Declination of first object [deg].
      ra2 : float
          Right ascension of second object [deg].
      dec2 : float
          Declination of second object [deg].

      Returns
      -------
      Position angle : float
      The position angle in degrees. the output will be
      given as an angle between 0 and 360 degrees.

    """

    # Convert into rad
    rarad1 = degtorad(ra1)
    rarad2 = degtorad(ra2)
    dcrad1 = degtorad(dec1)
    dcrad2 = degtorad(dec2)

    radif = rarad2 - rarad1

    angle = np.arctan2(np.sin(radif), np.cos(dcrad1) * np.tan(dcrad2) - np.sin(dcrad1) * np.cos(radif))

    result = radtodeg(angle)

    if positive and (result < 0.0):
        result += 360.0

    racen = (ra1 + ra2) / 2.0
    deccen = (dec1 + dec2) / 2.0

    if plot:
        plt.figure()
        ax1 = plt.subplot(111)
        ax1.set_xlim([racen + 1.5*abs(ra1-ra2), racen - 1.5*abs(ra1-ra2)])
        ax1.set_ylim([deccen - 1.5*abs(dec1-dec2), deccen + 1.5*abs(dec1-dec2)])

        ax1.plot(ra1,dec1,'o',color='darkorange',label='Target 1')
        ax1.plot(ra2,dec2,'bo',label='Target 2')

        ax1.annotate('',xy=(0.9,0.9),xycoords='axes fraction',xytext=(0.9, 0.695),
                    arrowprops=dict(arrowstyle="->",edgecolor='k',facecolor='k'),
                    horizontalalignment='left',verticalalignment='top')
        ax1.annotate('',xy=(0.7,0.7),xycoords='axes fraction',xytext=(0.905, 0.7),
                    arrowprops=dict(arrowstyle="->",edgecolor='k',facecolor='k'),
                    horizontalalignment='left',verticalalignment='top')

        ax1.text(ra1,dec1,'Target 1', fontsize=14, color='darkorange',horizontalalignment='left',verticalalignment='bottom')
        ax1.text(ra2,dec2,'Target 2', fontsize=14, color='b',horizontalalignment='left',verticalalignment='bottom')

        ax1.text(0.75,0.75,'E', fontsize=14, color='k',transform=ax1.transAxes)
        ax1.text(0.85,0.85,'N', fontsize=14, color='k',transform=ax1.transAxes)
        ax1.text(0.1,0.1,'PA={:0.2f}'.format(result), fontsize=14, color='k',transform=ax1.transAxes)
        plt.show()

    return np.round(result, 2)


def offset(ra1, dec1, ra2, dec2, center=False):
    """
    Compute the offset from object1 to object2 in ra and dec
    in units of arcsecond.

    Parameters
    ----------
    ra1 : float
    Right ascension of first object [deg].
    dec1 : float
    Declination of first object [deg].
    ra2 : float
    Right ascension of second object [deg].
    dec2 : float
    Declination of second object [deg].
    center: if true will also return the center position of these two objects

    Returns
    -------
    delta_ra,delta_dec : float
    center : float
    """

    # Convert into rad
    #rarad1 = degtorad(ra1)
    #rarad2 = degtorad(ra2)
    #dcrad1 = degtorad(dec1)
    #dcrad2 = degtorad(dec2)

    radif = (ra2 - ra1) * np.cos(degtorad((dec1 + dec2) / 2)) * 3600
    dcdif = (dec2 - dec1) * 3600

    if center:
        return np.round(radif, 2), np.round(dcdif, 2), (ra1 + ra2) / 2, (dec1 + dec2) / 2
    else:
        return np.round(radif, 2), np.round(dcdif, 2)
