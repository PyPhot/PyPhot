#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-
"""
This script enables the viewing of a raw FITS file
"""

def parse_args(options=None, return_parser=False):
    import argparse
    from pyphot.cameras import available_cameras

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('file', type = str, default = None, help = 'FITS file')
    parser.add_argument('camera', type=str,
                        help='A valid camera identifier: {0}'.format(
                             ', '.join(available_cameras)))
    parser.add_argument("--list", default=False, help="List the extensions only?", action="store_true")
    parser.add_argument('--exten', type=int, default = 0, help="FITS extension")
    parser.add_argument('--det', type=int, default=1, help="Detector number")

    if return_parser:
        return parser

    return parser.parse_args() if options is None else parser.parse_args(options)


def main(args):
    import subprocess
    from astropy.io import fits

    from pyphot.cameras import mmt_mmirs
    from pyphot.cameras import magellan_imacs


    # List only?
    if args.list:
        hdu = fits.open(args.file)
        print(hdu.info())
        return

    # RAW_MMIRS
    if args.camera == 'mmt_mmirs':
        gen_mmirs = mmt_mmirs.MMTMMIRSCamera()
        img = gen_mmirs.get_rawimage(args.file, args.det)[1]
    elif args.camera == 'magellan_imacsf2':
        gen_imacs = magellan_imacs.MagellanIMACSF2Camera()
        img = gen_imacs.get_rawimage(args.file, args.det)[1]
    else:
        hdu = fits.open(args.file)
        img = hdu[args.exten].data
        # Write

    #p = subprocess.Popen(["ds9", "-zscale"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #out, err = p.communicate()
