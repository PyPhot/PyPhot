import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord

ccd_shape = [2048,4096]
long_gap = 91
short_gap = 72


def plotbox(x0, x1, y0, y1, color='k', linestyle='-', fill_color='k', label=None, draw_frame=False):

    if draw_frame:
        plt.hlines(y0, x0, x1, linestyles=linestyle, colors=color)
        plt.hlines(y1, x0, x1, linestyles=linestyle, colors=color)
        plt.vlines(x0, y0, y1, linestyles=linestyle, colors=color)
        plt.vlines(x1, y0, y1, linestyles=linestyle, colors=color)
    plt.fill_between([x0,x1],[y0,y0],[y1,y1],color=fill_color,alpha=0.1)
    plt.text((x0+x1)/2,(y0+y1)/2,label, horizontalalignment='center', verticalalignment='center')
    print(x0, x1, y0, y1)


def one_mosaic(ccd_shape, long_gap, short_gap, xcen=0, ycen=0, color='k', linestyle='-', fill_color='k', show=False):

    ## ccd1
    x10 = xcen - 2*ccd_shape[0] - 3/2*long_gap
    x11 = x10 + ccd_shape[0]
    y10 = ycen + 1/2*short_gap
    y11 = y10 + ccd_shape[1]
    plotbox(x10,x11,y10,y11, color=color, linestyle=linestyle, fill_color=fill_color, label='CCD1')

    ## ccd2
    x20 = xcen - ccd_shape[0] - 1/2*long_gap
    x21 = x20 + ccd_shape[0]
    y20, y21 = y10, y11
    plotbox(x20,x21,y20,y21, color=color, linestyle=linestyle, fill_color=fill_color, label='CCD2')
    if xcen==0 and ycen==0:
        plt.plot(x21 - 537, y20 + 60, 'o', ms=5,linewidth=5, zorder=200, color='red')

    ## ccd3
    x30 = xcen + 1/2*long_gap
    x31 = x30 + ccd_shape[0]
    y30 , y31 = y10, y11
    plotbox(x30,x31,y30,y31, color=color, linestyle=linestyle, fill_color=fill_color, label='CCD3')

    ## ccd4
    x40 = xcen + 3/2*long_gap + ccd_shape[0]
    x41 = x40 + ccd_shape[0]
    y40 , y41 = y10, y11
    plotbox(x40,x41,y40,y41, color=color, linestyle=linestyle, fill_color=fill_color, label='CCD4')

    ## ccd6
    x60, x61 = x10, x11
    y60 = ycen - 1/2*short_gap
    y61 = y60 - ccd_shape[1]
    plotbox(x60,x61,y60,y61, color=color, linestyle=linestyle, fill_color=fill_color, label='CCD6')

    ## ccd5
    x50, x51 = x20, x21
    y50, y51 = y60, y61
    plotbox(x50,x51,y50,y51, color=color, linestyle=linestyle, fill_color=fill_color, label='CCD5')

    ## ccd8
    x80, x81 = x30, x31
    y80, y81 = y60, y61
    plotbox(x80, x81 ,y80,y81, color=color, linestyle=linestyle, fill_color=fill_color, label='CCD8')

    ## ccd7
    x70, x71 = x40, x41
    y70, y71 = y60, y61
    plotbox(x70, x71 ,y70,y71, color=color, linestyle=linestyle, fill_color=fill_color, label='CCD7')

    plt.plot(xcen, ycen, '+', ms=5, linewidth=5, zorder=200, color='gold')

    if show:
        plt.show()

def get_pos(ra, dec, pixscale=0.2):

    scale = pixscale/3600.0
    #ra0, dec0 = np.copy(ra), np.copy(dec) # The target position
    racen, deccen = ra+537*scale, dec-90*scale # The center is about 537 pixel to the East and 90 pixels to the south
    ra1,dec1 = ra, dec
    ra2,dec2 = ra-100.2*scale, dec+270.4*scale
    ra3,dec3 = ra-200.4*scale, dec-80.2*scale
    ra4,dec4 = ra+100.2*scale, dec-160.4*scale
    ra5,dec5 = ra+200.4*scale, dec+190.2*scale

    ra6,dec6 = ra-300*scale, dec+350*scale
    ra7,dec7 = ra-400.33*scale, dec-240.33*scale
    ra8,dec8 = ra+300.33*scale, dec-320.67*scale
    ra9,dec9 = ra+400.67*scale, dec+430.33*scale
    ra10,dec10 = ra-500.67*scale, dec+510.67*scale

    ra11,dec11 = ra-670*scale, dec-120*scale
    ra12,dec12 = ra-770*scale, dec+390*scale


    c = SkyCoord(ra=[ra1,ra2,ra3,ra4,ra5,ra6,ra7,ra8,ra9,ra10,ra11,ra12]*u.degree,
                 dec=[dec1,dec2,dec3,dec4,dec5,dec6,dec7,dec8,dec9,dec10,dec11,dec12]* u.degree, frame='icrs')
    for ipos in range(12):
        print(c.ra[ipos].deg, c.dec[ipos].deg, str(ipos+1))
        plt.plot(c.ra[ipos].deg, c.dec[ipos].deg,'r.') # target position
        plt.text(c.ra[ipos].deg, c.dec[ipos].deg, str(ipos + 1), va='center', ha='center', color='r')
        plt.plot(c.ra[ipos].deg+537*scale, c.dec[ipos].deg-90*scale, 'k+') # center position
        plt.text(c.ra[ipos].deg+537*scale, c.dec[ipos].deg-90*scale, str(ipos + 1), va='center', ha='center', color='k')
    #plt.plot(ra0, dec0, 'r+')
    plt.vlines(-4096*scale+ra, -4096*scale+dec, 4096*scale+dec)
    plt.vlines(4096*scale+ra, -4096*scale+dec, 4096*scale+dec)
    plt.hlines(-4096*scale+dec, -4096*scale+ra, 4096*scale+ra)
    plt.hlines(4096*scale+dec, -4096*scale+ra, 4096*scale+ra)
    plt.show()
    pos = c.to_string('hmsdms')
    for ipos in pos:
        print(ipos.replace('h',':').replace('m',':').replace('s','').replace('d',':'))


def plotdither(ndither=12):


    one_mosaic(ccd_shape, long_gap, short_gap, xcen=0, ycen=0, color='k', linestyle='-', fill_color='k', show=False)
    one_mosaic(ccd_shape, long_gap, short_gap, xcen=-100.2, ycen=270.4, color='r', linestyle='-', fill_color='k', show=False)
    one_mosaic(ccd_shape, long_gap, short_gap, xcen=-200.4, ycen=-80.2, color='b', linestyle='-', fill_color='k', show=False)
    one_mosaic(ccd_shape, long_gap, short_gap, xcen=100.2, ycen=-160.4, color='c', linestyle='-', fill_color='k', show=False)
    one_mosaic(ccd_shape, long_gap, short_gap, xcen=200.4, ycen=190.2, color='m', linestyle='-', fill_color='k', show=False)

    one_mosaic(ccd_shape, long_gap, short_gap, xcen=-300, ycen=350, color='y', linestyle='-', fill_color='k', show=False)
    one_mosaic(ccd_shape, long_gap, short_gap, xcen=-400.33, ycen=-240.33, color='navy', linestyle='-', fill_color='k', show=False)
    one_mosaic(ccd_shape, long_gap, short_gap, xcen=300.33, ycen=-320.67, color='brown', linestyle='-', fill_color='k', show=False)
    one_mosaic(ccd_shape, long_gap, short_gap, xcen=400.67, ycen=430.33, color='dodgerblue', linestyle='-', fill_color='k', show=False)
    one_mosaic(ccd_shape, long_gap, short_gap, xcen=-500.67, ycen=510.67, color='orange', linestyle='-', fill_color='k', show=False)

    one_mosaic(ccd_shape, long_gap, short_gap, xcen=-670, ycen=-120, color='g', linestyle='-', fill_color='k', show=False)
    one_mosaic(ccd_shape, long_gap, short_gap, xcen=-770, ycen=390, color='dodgerblue', linestyle='-', fill_color='k', show=True)

#get_pos(231.6576708, -20.8335167)
#plotdither()
#get_pos(323.13829166666665, 12.298683333333333)
#get_pos(36.1105833,-47.1915000)
#get_pos(36.50779166666667, 3.0498)
#get_pos(81.4986458,-24.1063833)
#get_pos(140.3356667,0.1230278)
#get_pos(140.41900003333333,0.1230278) # shift 5 arcmin to the east to avoid bright star
#get_pos(41.00425,-50.14825)
#get_pos(67.68191666666665,-14.761447222222221)
#get_pos(140.9958208333333,7.896974166666666)
#get_pos(167.64149999999998,-13.495999999999999)
#get_pos(300.67330833333335,-30.222691666666666)
#get_pos(140.4356667,0.1230278) # shift 6 arcmin to the east to avoid bright star

'''
one_mosaic(ccd_shape, long_gap, short_gap, xcen=-50, ycen=-400, color='dodgerblue', linestyle='-', fill_color='k', show=False)
one_mosaic(ccd_shape, long_gap, short_gap, xcen=250, ycen=-50, color='g', linestyle='-', fill_color='k', show=False)
one_mosaic(ccd_shape, long_gap, short_gap, xcen=50, ycen=350, color='lime', linestyle='-', fill_color='k', show=False)
one_mosaic(ccd_shape, long_gap, short_gap, xcen=-350, ycen=30, color='pink', linestyle='-', fill_color='k', show=False)

one_mosaic(ccd_shape, long_gap, short_gap, xcen=-650, ycen=-250, color='coral', linestyle='-', fill_color='k', show=False)
one_mosaic(ccd_shape, long_gap, short_gap, xcen=-750, ycen=180, color='skyblue', linestyle='-', fill_color='k', show=False)
one_mosaic(ccd_shape, long_gap, short_gap, xcen=-500, ycen=500, color='orange', linestyle='-', fill_color='k', show=True)
'''