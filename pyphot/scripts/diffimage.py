#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-
"""
"""

def parse_args(options=None, return_parser=False):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('fileA', type = str, default = None, help = 'FITS file A')
    parser.add_argument('fileB', type = str, default = None, help = 'FITS file B')
    parser.add_argument('--exten', type=int, default = 0, help="FITS extension")
    parser.add_argument('--outfile', type=str, default = 'diffAB.fits', help="Outout fits file name")
    parser.add_argument('--show', type=bool, default = True, help="Show the difference image with ds9?")

    if return_parser:
        return parser

    return parser.parse_args() if options is None else parser.parse_args(options)


def main(args):

    import os
    from pyphot import msgs
    from astropy.io import fits

    parA = fits.open(args.fileA)
    parB = fits.open(args.fileB)

    data = parA[args.exten].data*1.0 - parB[args.exten].data*1.0

    parA[args.exten].data = data

    outfile = args.outfile
    if outfile.split('.')[-1] != 'fits':
        outfile = outfile+'.fits'
    msgs.info('Writing to {:}'.format(outfile))
    parA.writeto(outfile,overwrite=True)

    #if args.show:
    #    from pypeit.display import display
    #    display.show_image(outfile, chname="Diff_IMG", wcs_match=True)



