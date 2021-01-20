#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-
"""
This script runs PyPhot
"""

from pyphot import msgs

def run_pyphot_usage():
    """
    Print pyphot usage description.
    """

    import textwrap
    import pyphot
    from pyphot.cameras import available_cameras

    spclist = ', '.join(available_cameras)
    spcl = textwrap.wrap(spclist, width=70)
    descs = '##  '
    descs += '\x1B[1;37;42m' + 'PyPhot : '
    descs += 'The Python Image Data Reduction Pipeline v{0:s}'.format(pyphot.__version__) \
              + '\x1B[' + '0m' + '\n'
    descs += '##  '
    descs += '\n##  Available cameras include:'
    for ispcl in spcl:
        descs += '\n##   ' + ispcl
    return descs


def parse_args(options=None, return_parser=False):
    import argparse

    parser = argparse.ArgumentParser(description=run_pyphot_usage(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pyphot_file', type=str,
                        help='PyPhot reduction file (must have .pyphot extension)')
    parser.add_argument('-v', '--verbosity', type=int, default=2,
                        help='Verbosity level between 0 [none] and 2 [all]')
    parser.add_argument('-r', '--redux_path', default=None,
                        help='Path to directory for the reduction.  Only advised for testing')
    parser.add_argument('-m', '--do_not_reuse_masters', default=False, action='store_true',
                        help='Do not load previously generated MasterFrames, even ones made during the run.')
    parser.add_argument('-s', '--show', default=False, action='store_true',
                        help='Show reduction steps via plots (which will block further execution until clicked on) '
                             'and outputs to ginga. Requires remote control ginga session via "ginga --modules=RC &"')
    # JFH Should the default now be true with the new definition.
    parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                        help='Overwrite any existing files/directories')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--detector', default=None, help='Detector to limit reductions on.  If the output files exist and -o is used, the outputs for the input detector will be replaced.')
    parser.add_argument('-c', '--calib_only', default=False, action='store_true',
                         help='Only run on calibrations')

    if return_parser:
        return parser

    return parser.parse_args() if options is None else parser.parse_args(options)


def main(args):

    import os

    from pyphot import pyphot

    # Initiate logging for bugs and command line help
    # These messages will not be saved to a log file
    # Set the default variables
    qck = False
    cpu = 1
    #vrb = 2

    # Load options from command line
    splitnm = os.path.splitext(args.pyphot_file)
    if splitnm[1] != '.pyphot':
        msgs.error("Bad extension for PyPhot reduction file."+msgs.newline()+".pyphot is required")
    logname = splitnm[0] + ".log"

    # Instantiate the main pipeline reduction object
    Pyphot = pyphot.PyPhot(args.pyphot_file, verbosity=args.verbosity,
                           reuse_masters=~args.do_not_reuse_masters,
                           overwrite=args.overwrite,
                           redux_path=args.redux_path,
                           calib_only=args.calib_only,
                           logname=logname, show=args.show)

    # JFH I don't see why this is an optional argument here. We could allow the user to modify an infinite number of parameters
    # from the command line? Why do we have the PyPhot file then? This detector can be set in the pyphot file.
    # Detector?
    if args.detector is not None:
        msgs.info("Restricting reductions to detector={}".format(args.detector))
        Pyphot.par['rdx']['detnum'] = int(args.detector)

    if args.calib_only:
        Pyphot.calib_all()
    else:
        Pyphot.reduce_all()
    msgs.info('Data reduction complete')
    # QA HTML
    msgs.info('Generating QA HTML')
    Pyphot.build_qa()

    return 0

