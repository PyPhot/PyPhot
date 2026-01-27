"""This module contains the tools needed for satellite detection
It was originally written by David Borncamp and  Pey Lian Lim

#
# History:
#    Dec 12, 2014 - DMB - Created for COSC 602 "Image Processing and Pattern
#        Recocnition" at Towson University. Mostly detection algorithm
#        development and VERY crude mask.
#    Feb 01, 2015 - DMB - Masking algorithm refined to be useable by HFF.
#    Mar 28, 2015 - DMB - Small bug fixes and tweaking to try not to include
#        diffraction spikes.
#    Nov 03, 2015 - PLL - Adapted for acstools distribution. Fixed bugs,
#        possibly improved performance, changed API.
#    Dec 07, 2015 - PLL - Minor changes based on feedback from DMB.
#    May 24, 2016 - SMO - Minor import changes to skimage
#
#    Jul 14, 2021 - Import to PyPhot by FW.
"""

import numpy as np

from skimage import transform
from skimage import morphology as morph
from skimage import exposure
from skimage.feature import canny

from pyphot import msgs

def satdet(image, bpm=None, sigma=3.0, buf=20, order=3, low_thresh=0.1, h_thresh=0.5,
           small_edge=60, line_len=200, line_gap=75, percentile=(4.5, 93.0), verbose=True):

    # rescale the image
    p1, p2 = np.percentile(image, percentile)

    # there should always be some counts in the image, anything lower should
    # be set to one. Makes things nicer for finding edges.
    if p1 < 0:
        p1 = 0.0

    # rescale the image
    if bpm is None:
        bpm = image==0.
    image = exposure.rescale_intensity(image, in_range=(p1, p2))

    # get the edges
    immax = np.max(image)
    edge = canny(image, sigma=sigma,
                 low_threshold=immax * low_thresh,
                 high_threshold=immax * h_thresh)
    edge[bpm] = 0.

    # clean up the small objects, will make less noise
    morph.remove_small_objects(edge, min_size=small_edge, connectivity=8,
                               in_place=True)

    # create an array of angles from 0 to 180, exactly 0 will get bad columns
    # but it is unlikely that a satellite will be exactly at 0 degrees, so
    # don't bother checking.
    # then, convert to radians.
    angle = np.radians(np.arange(1, 179, 0.5, dtype=float))

    # perform Probabilistic Hough Transformation to get line segments.
    # NOTE: Results are slightly different from run to run!
    result = transform.probabilistic_hough_line(
        edge, threshold=210, line_length=line_len,
        line_gap=line_gap, theta=angle)
    result = np.asarray(result)
    n_result = int(np.size(result)/4) # each line has four elements

    #import matplotlib.pyplot as plt
    #plt.imshow(edge, origin='lower')
    #for ii in range(n_result):
    #    x0, y0 = result[ii][0]
    #    x1, y1 = result[ii][1]
    #    #plt.plot(x0,y0,'b-')
    #    #plt.plot(x1,y1,'r-')
    #    plt.plot([x0,x1],[y0,y1],'--',lw=1)
    #plt.show()

    if n_result <=1:
        SATELLITE=False
    else:
        # create lists for X and Y positions of lines and build points
        x0 = result[:, 0, 0]
        y0 = result[:, 0, 1]
        x1 = result[:, 1, 0]
        y1 = result[:, 1, 1]

        # set up trail angle "tracking" arrays.
        # find the angle of each segment and filter things out.
        # TODO: this may be wrong. Try using arctan2.
        trail_angle = np.degrees(np.arctan((y1 - y0) / (x1 - x0)))
        # round to the nearest 5 degrees, trail should not be that curved
        round_angle = (5 * np.round(trail_angle * 0.2)).astype(int)

        # remove lines along column or line
        # since it is unlikely that a satellite will be exactly at 0 or 90 degrees
        # take out 90 degree things
        mask = round_angle % 90 != 0

        if np.sum(mask)<=1:
            SATELLITE = False
        else:
            round_angle = round_angle[mask]
            trail_angle = trail_angle[mask]
            result = result[mask]

            # identify trail groups and reject those only with one line
            # The original code can only identify satellite trails at the same angle
            # and only counts for those trails traversed the image
            # I changed the algorithm to identify all trails in the image no matter whether
            # the trail traverses the image or not.
            angle_grp = np.unique(round_angle)
            nline_grp = np.zeros_like(angle_grp)
            for igrp, iangle in enumerate(angle_grp):
                nline_grp[igrp] = np.sum(round_angle==iangle)
            final_grp_angle = angle_grp[nline_grp>1]
            final_grp_nline = nline_grp[nline_grp>1]


            if len(final_grp_angle)<1:
                SATELLITE = False
            else:
                if verbose:
                    msgs.info('Identifyied {:} groups of satellite trails.'.format(len(final_grp_angle)))
                n_satellite = 0
                mask_final = np.zeros_like(image, dtype='bool')
                ymax, xmax = np.shape(image)
                topx = xmax - buf
                topy = ymax - buf

                for igrp, iangle in enumerate(final_grp_angle):
                    this_angle = trail_angle[round_angle==iangle]
                    this_result = result[round_angle==iangle]
                    this_nresult = int(np.size(this_result) / 4)  # each line has four elements
                    this_nline = final_grp_nline[igrp]

                    # Determine whether the trails traversed the image
                    traversed=False
                    this_x0 = this_result[:, 0, 0]
                    this_y0 = this_result[:, 0, 1]
                    this_x1 = this_result[:, 1, 0]
                    this_y1 = this_result[:, 1, 1]

                    min_x0, min_x1 = min(this_x0), min(this_x1)
                    min_y0, min_y1 = min(this_y0), min(this_y1)
                    max_x0, max_x1 = max(this_x0), max(this_x1)
                    max_y0, max_y1 = max(this_y0), max(this_y1)

                    # top-bottom
                    if (min_y0 < buf) or (min_y1 < buf) or (max_y0 > topy) or (max_y1 > topy):
                        traversed = True
                    # left-right
                    if (min_x0 < buf) or (min_x1 < buf) or (max_x0 > topx) or (max_x1 > topx):
                        traversed = True

                    ## rotate the image and identify how many trails in each group
                    this_deg = np.median(this_angle)
                    this_rad = np.radians(this_deg)
                    point_rad = 2 * np.pi - this_rad # counter-clock

                    # rotate the image
                    this_img_rotate = transform.rotate(image, this_deg, resize=True, order=order)
                    this_mask_rotate = np.zeros_like(this_img_rotate, dtype='bool')

                    # get the indices of line coordinates in the rotated image
                    #plt.imshow(this_img_rotate, origin='lower')
                    this_result_rotate = np.zeros_like(this_result)
                    for ii in range(this_nresult):
                        rx0, ry0 = _rotate_point((this_result[ii, 0, 0], this_result[ii, 0, 1]),
                                                 point_rad, image.shape, this_img_rotate.shape)
                        rx1, ry1 = _rotate_point((this_result[ii, 1, 0], this_result[ii, 1, 1]),
                                                 point_rad, image.shape, this_img_rotate.shape)
                        # since I used a median angle for rotating the image, its possible that ry0!=ry1 for some cases
                        # thus I compute the mean of ry0 and ry1
                        ry = int(np.mean([ry0,ry1]))
                        this_result_rotate[ii, 0, 0] = rx0
                        this_result_rotate[ii, 0, 1] = ry
                        this_result_rotate[ii, 1, 0] = rx1
                        this_result_rotate[ii, 1, 1] = ry
                        #plt.plot(rx0, ry, 'bo', lw=1)
                        #plt.plot(rx1, ry, 'ro', lw=1)
                    #plt.show()

                    # figure out how many trails in each group
                    if this_nline<=3:
                        if verbose:
                            msgs.info('Identified one satellite trail at angle={:} degree.'.format(iangle))
                        xx = np.hstack([this_result_rotate[:, 0, 0],this_result_rotate[:, 1, 0]])
                        yy = np.hstack([this_result_rotate[:, 0, 1],this_result_rotate[:, 1, 1]])
                        if traversed:
                            x_low, x_high = 0, np.shape(this_img_rotate)[1]
                        else:
                            x_low, x_high = np.min(xx), np.max(xx)
                        y_low, y_high = np.min(yy), np.max(yy)
                        this_mask_rotate[y_low:y_high+1, x_low:x_high+1] = True
                        n_satellite +=1
                    else:
                        # ToDo: Currently I assume that one trail per group.
                        # and simply copyed the above codes here.
                        if verbose:
                            msgs.info('Identified one satellite trail at angle={:} degree.'.format(iangle))
                        xx = np.hstack([this_result_rotate[:, 0, 0],this_result_rotate[:, 1, 0]])
                        yy = np.hstack([this_result_rotate[:, 0, 1],this_result_rotate[:, 1, 1]])
                        if traversed:
                            # extend the trail to the whole row just in case the line point was missed at the image edge
                            x_low, x_high = 0, np.shape(this_img_rotate)[1]
                        else:
                            x_low, x_high = np.min(xx), np.max(xx)
                        y_low, y_high = np.min(yy), np.max(yy)
                        this_mask_rotate[y_low:y_high+1, x_low:x_high+1] = True
                        n_satellite +=1

                        ## ToDo: developing should start from here.
                        #msgs.work('TBD.')
                        #yy = this_result_rotate[:, 0, 1]
                        #this_sort = np.argsort(yy)
                        #yy_sort = yy[this_sort]
                        #this_result_rotate = this_result_rotate[this_sort,:,:]

                    ## rotate the mask back and return
                    this_mask = transform.rotate(this_mask_rotate, -this_deg, resize=True, order=order)

                    ix0 = (this_mask.shape[1] - image.shape[1]) / 2
                    iy0 = (this_mask.shape[0] - image.shape[0]) / 2
                    lowerx, upperx, lowery, uppery = _get_valid_indices(
                        this_mask.shape, ix0, image.shape[1] + ix0, iy0, image.shape[0] + iy0)
                    this_mask_new = this_mask[lowery:uppery, lowerx:upperx]
                    mask_final += this_mask_new.astype(bool)

                if n_satellite>0:
                    SATELLITE = True
                else:
                    SATELLITE = False

    if SATELLITE:
        if verbose:
            msgs.info('Identified {:} satellite trails'.format(n_satellite))
        return mask_final
    else:
        if verbose:
            msgs.info('No satellite trail was identified.')
        return np.zeros_like(image, dtype='bool')

def _get_valid_indices(shape, ix0, ix1, iy0, iy1):
    """Give array shape and desired indices, return indices that are
    correctly bounded by the shape."""
    ymax, xmax = shape

    if ix0 < 0:
        ix0 = 0
    if ix1 > xmax:
        ix1 = xmax
    if iy0 < 0:
        iy0 = 0
    if iy1 > ymax:
        iy1 = ymax

    if iy1 <= iy0 or ix1 <= ix0:
        raise IndexError(f'array[{iy0}:{iy1},{ix0}:{ix1}] is invalid')

    return list(map(int, [ix0, ix1, iy0, iy1]))


def _rotate_point(point, angle, ishape, rshape, reverse=False):
    """Transform a point from original image coordinates to rotated image
    coordinates and back. It assumes the rotation point is the center of an
    image.

    This works on a simple rotation transformation::

        newx = (startx) * np.cos(angle) - (starty) * np.sin(angle)
        newy = (startx) * np.sin(angle) + (starty) * np.cos(angle)

    It takes into account the differences in image size.

    Parameters
    ----------
    point : tuple
        Point to be rotated, in the format of ``(x, y)`` measured from
        origin.

    angle : float
        The angle in degrees to rotate the point by as measured
        counter-clockwise from the X axis.

    ishape : tuple
        The shape of the original image, taken from ``image.shape``.

    rshape : tuple
        The shape of the rotated image, in the form of ``rotate.shape``.

    reverse : bool, optional
        Transform from rotated coordinates back to non-rotated image.

    Returns
    -------
    rotated_point : tuple
        Rotated point in the format of ``(x, y)`` as measured from origin.

    """
    #  unpack the image and rotated images shapes
    if reverse:
        angle = (angle * -1)
        temp = ishape
        ishape = rshape
        rshape = temp

    # transform into center of image coordinates
    yhalf, xhalf = ishape
    yrhalf, xrhalf = rshape

    yhalf = yhalf / 2
    xhalf = xhalf / 2
    yrhalf = yrhalf / 2
    xrhalf = xrhalf / 2

    startx = point[0] - xhalf
    starty = point[1] - yhalf

    # do the rotation
    newx = startx * np.cos(angle) - starty * np.sin(angle)
    newy = startx * np.sin(angle) + starty * np.cos(angle)

    # add back the padding from changing the size of the image
    newx = newx + xrhalf
    newy = newy + yrhalf

    return (newx, newy)
