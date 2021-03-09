#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-
"""
This script generates files to setup a PyPhot run
"""
import os
from pyphot.cameras import available_cameras
from pyphot.pyphotsetup import PyPhotSetup


def parse_args(options=None, return_parser=False):
    import argparse

    # TODO: Add argument that specifies the log file
    parser = argparse.ArgumentParser(description='Parse data files to construct a pyphot file in '
                                                 'preparation for reduction using \'run_pyphot\'',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TODO: Make root and spectrograph required arguments
    parser.add_argument('-r', '--root', type=str, default=None,
                       help='File path+root, e.g. /data/Kast/b ')
    parser.add_argument('-c', '--camera', default=None, type=str,
                        help='A valid camera identifier: {0}'.format(
                                ', '.join(available_cameras)))

    parser.add_argument('-e', '--extension', default='.fits',
                        help='File extension; compression indicators (e.g. .gz) not required.')
    parser.add_argument('-d', '--output_path', default=os.getcwd(),
                        help='Path to top-level output directory.')
    parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                        help='Overwrite any existing files/directories')
    parser.add_argument('-s', '--cfg_split', default=None, type=str,
                        help='Generate the PyPhot files and folders by input configuration.  To '
                             'write all unique configurations identifed, use \'all\', otherwise '
                             'provide the list of configuration letters; e.g., \'A,B\' or '
                             '\'B,D,E\' or \'E\'.')
    parser.add_argument('-v', '--verbosity', type=int, default=2,
                        help='Level of verbosity from 0 to 2.')

    if return_parser:
        return parser

    return parser.parse_args() if options is None else parser.parse_args(options)


def main(args):


    if args.root is None:
        raise IOError('root is a required argument.  Use the -r, --root command-line option.')
    if args.camera is None:
        raise IOError('camera is a required argument.  Use the -c, --camera '
                      'command-line option.')

    # Check that input spectrograph is supported
    if args.camera not in available_cameras:
        raise ValueError('Instrument \'{0}\' unknown to PyPhot.\n'.format(args.camera)
                         + '\tOptions are: {0}\n'.format(', '.join(available_cameras))
                         + '\tSelect an available instrument or consult the documentation '
                         + 'on how to add a new instrument.')

    # Get the output directory
    sort_dir = os.path.join(args.output_path, 'setup_files')

    # Initialize PyPhotSetup based on the arguments
    ps = PyPhotSetup.from_file_root(args.root, args.camera, extension=args.extension,
                                    output_path=sort_dir)
    # Run the setup
    ps.run(setup_only=True, sort_dir=sort_dir)

    # Use PyPhotMetaData to write the complete PyPhot file
    if args.cfg_split is not None:
        ps.fitstbl.write_pyphot(args.output_path, cfg_lines=ps.user_cfg,
                                configs=[item.strip() for item in args.cfg_split.split(',')])

