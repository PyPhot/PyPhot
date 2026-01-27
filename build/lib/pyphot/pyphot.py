"""
Main driver class for PyPhot run

Modified from PyPeIt.
"""
import time
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table

from configobj import ConfigObj
from collections import OrderedDict

from pyphot import msgs, io, utils
from pyphot import procimg, postproc
from pyphot.par.util import parse_pyphot_file
from pyphot.par import PyPhotPar
from pyphot.metadata import PyPhotMetaData
from pyphot.cameras.util import load_camera
from pyphot import masterframe
from pyphot.psf import  psf


class PyPhot(object):
    """
    This class runs the primary calibration and extraction in PyPhot

    .. todo::
        Fill in list of attributes!

    Args:
        pyphot_file (:obj:`str`):
            PyPhot filename.
        verbosity (:obj:`int`, optional):
            Verbosity level of system output.  Can be:

                - 0: No output
                - 1: Minimal output (default)
                - 2: All output

        overwrite (:obj:`bool`, optional):
            Flag to overwrite any existing files/directories.
        reuse_masters (:obj:`bool`, optional):
            Reuse any pre-existing calibration files
        logname (:obj:`str`, optional):
            The name of an ascii log file with the details of the
            reduction.
        redux_path (:obj:`str`, optional):
            Over-ride reduction path in PyPhot file (e.g. Notebook usage)

    Attributes:
        pyphot_file (:obj:`str`):
            Name of the pyphot file to read.  PyPhot files have a
            specific set of valid formats. A description can be found
            :ref:`pyphot_file`.
        fitstbl (:obj:`pyphot.metadata.PyPhotMetaData`): holds the meta info

    """

    def __init__(self, pyphot_file, verbosity=2, overwrite=True, reuse_masters=False, logname=None,
                 redux_path=None):

        # Set up logging
        self.logname = logname
        self.verbosity = verbosity
        self.pyphot_file = pyphot_file
        
        self.msgs_reset()

        # Load
        cfg_lines, data_files, frametype, usrdata, setups \
                = parse_pyphot_file(pyphot_file, runtime=True)

        # Spectrograph
        cfg = ConfigObj(cfg_lines)
        camera_name = cfg['rdx']['camera']
        self.camera = load_camera(camera_name)
        msgs.info('Loaded camera {0}'.format(self.camera.name))

        # --------------------------------------------------------------
        # Get the full set of PyPhot parameters
        #   - Grab a science or standard file for configuration specific parameters

        config_specific_file = None
        for idx, row in enumerate(usrdata):
            if ('science' in row['frametype']) or ('standard' in row['frametype']):
                config_specific_file = data_files[idx]
        if config_specific_file is not None:
            msgs.info(
                'Setting configuration-specific parameters using {0}'.format(os.path.split(config_specific_file)[1]))
        camera_cfg_lines = self.camera.config_specific_par(config_specific_file).to_config()

        #   - Build the full set, merging with any user-provided
        #     parameters
        self.par = PyPhotPar.from_cfg_lines(cfg_lines=camera_cfg_lines, merge_with=cfg_lines)
        msgs.info('Built full PyPhot parameter set.')

        # Check the output paths are ready
        if redux_path is not None:
            self.par['rdx']['redux_path'] = redux_path

        # TODO: Write the full parameter set here?
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # Build the meta data
        #   - Re-initilize based on the file data
        msgs.info('Compiling metadata')
        self.fitstbl = PyPhotMetaData(self.camera, self.par, files=data_files,
                                      usrdata=usrdata, strict=True)
        #   - Interpret automated or user-provided data from the PyPhot
        #   file
        self.fitstbl.finalize_usr_build(frametype, setups[0])

        # Other Internals
        self.overwrite = overwrite

        # Currently the runtime argument determines the behavior for
        # reuse_masters.
        self.reuse_masters = reuse_masters

        # Set paths
        self.calibrations_path = os.path.join(self.par['rdx']['redux_path'], self.par['calibrations']['master_dir'])

        if self.qa_path is not None and not os.path.isdir(self.qa_path):
            os.makedirs(self.qa_path)
        if self.calibrations_path is not None and not os.path.isdir(self.calibrations_path):
            os.makedirs(self.calibrations_path)
        if self.science_path is not None and not os.path.isdir(self.science_path):
            os.makedirs(self.science_path)
        if self.coadd_path is not None and not os.path.isdir(self.coadd_path):
            os.makedirs(self.coadd_path)

        # Report paths
        msgs.info('Setting reduction path to {0}'.format(self.par['rdx']['redux_path']))
        msgs.info('Quality assessment plots output to: {0}'.format(self.qa_path))
        msgs.info('Master calibration data output to: {0}'.format(self.calibrations_path))
        msgs.info('Science data output to: {0}'.format(self.science_path))
        msgs.info('Coadded data output to: {0}'.format(self.coadd_path))

        # Init

        self.tstart = None
        self.basename = None
        self.sciI = None
        self.obstime = None

    @property
    def science_path(self):
        """Return the path to the science directory."""
        return os.path.join(self.par['rdx']['redux_path'], self.par['rdx']['scidir'])

    @property
    def qa_path(self):
        """Return the path to the top-level QA directory."""
        return os.path.join(self.par['rdx']['redux_path'], self.par['rdx']['qadir'])

    @property
    def coadd_path(self):
        """Return the path to the top-level QA directory."""
        return os.path.join(self.par['rdx']['redux_path'], self.par['rdx']['coadddir'])

    def reduce_all(self):
        """
        Main driver of the entire reduction

        Calibration and extraction via a series of calls to reduce_exposure()

        """
        # Validate the parameter set
        self.par.validate_keys(required=['rdx', 'calibrations', 'scienceframe', 'postproc'])
        self.tstart = time.time()

        # Find the standard frames
        is_standard = self.fitstbl.find_frames('standard')

        # Find the bias frames
        is_bias = self.fitstbl.find_frames('bias')

        # Find the dark frames
        is_dark = self.fitstbl.find_frames('dark')

        # Find the pixel flat frames
        is_pixflat = self.fitstbl.find_frames('pixelflat')

        # Find the illuminate flat frames
        is_illumflat = self.fitstbl.find_frames('illumflat')

        # Find the super sky frames
        is_supersky = self.fitstbl.find_frames('supersky')

        # Find the fringe frames
        is_fringe = self.fitstbl.find_frames('fringe')

        # Find the science frames
        is_science = self.fitstbl.find_frames('science')

        # Frame indices
        frame_indx = np.arange(len(self.fitstbl))

        # Find the detectors to reduce
        detectors = PyPhot.select_detectors(detnum=self.par['rdx']['detnum'],
                                            ndet=self.camera.ndet)

        ## Start data processing
        # Steo one: build master calibrations
        # Step two: Imaging processing
        #           - detproc, gain correction, bias, dark, flat
        #           - sciproc, supersky flattening, background subtraction, defringing
        # Step three: Post processing
        #           - astrometry
        #           - calibrate zeropoint for individual chips
        #           - produce image QA for individual chips
        #           - Coadd images according to coadd_ids
        #           - Generate source catalogs

        ## Step one and two are iterate over n_calib_groups
        for i in range(self.fitstbl.n_calib_groups):
            # Find all the frames in this calibration group
            in_grp = self.fitstbl.find_calib_group(i)
            in_grp_sci = is_science & in_grp
            in_grp_supersky = is_supersky & in_grp
            in_grp_fringe = is_fringe & in_grp

            if np.sum(in_grp)<1:
                msgs.info('No frames found for the {:}th calibration group, skipping.'.format(i))
            else:
                this_setup = self.fitstbl[in_grp]['setup'][0]
                # Find the indices of the science frames in this calibration group:
                grp_all = frame_indx[in_grp] # science only
                grp_science = frame_indx[in_grp_sci] # science only
                grp_proc = frame_indx[in_grp_sci | in_grp_supersky | in_grp_fringe] # need run detproc
                grp_supersky = frame_indx[in_grp_supersky] # supersky
                grp_fringe = frame_indx[in_grp_fringe] # fringe
                grp_sciproc = frame_indx[in_grp_sci | in_grp_fringe] # need run both detproc and sciproc

                allfiles = self.fitstbl.frame_paths(grp_all)  # list for all files in this grp

                # science file lists
                scifiles = self.fitstbl.frame_paths(grp_science)  # list for scifiles

                # calibration file lists
                grp_bias = frame_indx[is_bias & in_grp]
                biasfiles = self.fitstbl.frame_paths(grp_bias)

                grp_dark = frame_indx[is_dark & in_grp]
                darkfiles = self.fitstbl.frame_paths(grp_dark)

                grp_illumflat = frame_indx[is_illumflat & in_grp]
                illumflatfiles = self.fitstbl.frame_paths(grp_illumflat)

                grp_pixflat = frame_indx[is_pixflat & in_grp]
                pixflatfiles = self.fitstbl.frame_paths(grp_pixflat)

                superskyfiles = self.fitstbl.frame_paths(grp_supersky) # supersky files
                fringefiles = self.fitstbl.frame_paths(in_grp_fringe) # Fringe files

                # proc file lists
                procfiles = self.fitstbl.frame_paths(grp_proc)  # need run detproc
                proc_airmass = self.fitstbl[grp_proc]['airmass']

                sciprocfiles = self.fitstbl.frame_paths(grp_sciproc)  # need run both detproc and sciproc
                sciproc_airmass = self.fitstbl[grp_sciproc]['airmass']

                master_keys = []
                raw_shapes = []
                for ii, idet in enumerate(detectors):
                    master_key = self.fitstbl.master_key(grp_all[0], det=idet)
                    raw_shape = self.camera.bpm(allfiles[0], idet, shape=None, msbias=None).astype('bool').shape
                    master_keys.append(master_key)
                    raw_shapes.append(raw_shape)

                ## Build MasterFrames, including bias, dark, illumflat, and pixelflat
                if not self.par['rdx']['skip_master']:
                    masterframe.build_masters(detectors, master_keys, raw_shapes, camera=self.camera, par=self.par,
                                              biasfiles=biasfiles, darkfiles=darkfiles, pixflatfiles=pixflatfiles,
                                              illumflatfiles=illumflatfiles, reuse_masters=self.reuse_masters)

                    ## ToDo: Re-scale the Flat normalizations in different detectors. Is this correct?
                    if len(detectors)>1:
                        masterframe.rescale_flat(self.camera, self.par, detectors, master_keys, raw_shapes)

                if np.sum(in_grp_sci) > 0:
                    ## Data processing, including detproc and sciproc
                    ## Loop over detectors for detproc and sciproc
                    for ii, idet in enumerate(detectors):
                        master_key = master_keys[ii]
                        raw_shape = raw_shapes[ii]
                        ## Load master files
                        Master = masterframe.MasterFrames(self.par, self.camera, idet, master_key, raw_shape,
                                                          reuse_masters=self.reuse_masters)
                        masterbiasimg, masterdarkimg, masterillumflatimg, masterpixflatimg, bpm_proc,\
                            norm_illum, norm_pixel = Master.load()

                        ## Initialize ImageProc
                        Proc = procimg.ImageProc(self.par, self.camera, idet, self.science_path, master_key, raw_shape,
                                                 reuse_masters=self.reuse_masters, overwrite=self.overwrite)

                        ## DETPROC -- bias, dark subtraction and flat fielding, support parallel processing
                        if not self.par['rdx']['skip_detproc']:
                            Proc.run_detproc(procfiles, masterbiasimg, masterdarkimg,
                                             masterpixflatimg, masterillumflatimg, bpm_proc)

                        ## SCIPROC -- supersky flattening, extinction correction based on airmass, and background subtraction.
                        if not self.par['rdx']['skip_sciproc']:

                            ## Build SuperSkyFlat first
                            if self.par['scienceframe']['process']['use_supersky']:
                                Proc.build_supersky(superskyfiles)

                            ## Run sciproc
                            Proc.run_sciproc(sciprocfiles, sciproc_airmass)

                            ## Build Master Fringing and Defringing
                            if self.par['scienceframe']['process']['use_fringe']:
                                Proc.build_fringe(fringefiles)
                                # Defringing
                                Proc.run_defringing(scifiles)
                else:
                    msgs.info('No science images for the {:}th calibration group.'.format(i))

        ## Step three is iterated over coadd_ids
        # prepare some useful lists
        scifiles = self.fitstbl.frame_paths(is_science)  # list for scifiles
        sci_airmass = self.fitstbl['airmass'][is_science]
        sci_exptime = self.fitstbl['exptime'][is_science]
        sci_filter = self.fitstbl['filter'][is_science]
        sci_target = self.fitstbl['target'][is_science]
        sci_ra = self.fitstbl['ra'][is_science]
        sci_dec = self.fitstbl['dec'][is_science]
        coadd_ids = self.fitstbl['coadd_id'][is_science]  # coadd_ids

        # determine median pixel scale for the detectors that will be used for astrometric calibration
        pixscales = []
        for idet in detectors:
            detector_par = self.camera.get_detector_par(fits.open(scifiles[0]), idet)
            pixscales.append(detector_par['platescale'])
        pixscale = np.median(pixscales)

        # Initiallize PostProc
        Post = postproc.PostProc(self.par, detectors, this_setup, scifiles, coadd_ids, sci_ra, sci_dec, sci_airmass,
                                 sci_exptime, sci_filter, sci_target, pixscale, self.science_path,
                                 self.qa_path, self.coadd_path, overwrite=self.overwrite,
                                 reuse_masters=self.reuse_masters)

        # run astrometry
        if not self.par['rdx']['skip_astrometry']:
            Post.run_astrometry()

        # run chipcal
        if not self.par['rdx']['skip_chipcal']:
            Post.run_chip_cal()

        # Making QA image for calibrated individual chips
        if not self.par['rdx']['skip_img_qa']:
            Post.run_img_qa()

        # Run coadd
        if not self.par['rdx']['skip_coadd']:
            Post.run_coadd()

        # Extract photometric catalog
        if not self.par['rdx']['skip_detection']:
            Post.extract_catalog()

        # Finish
        self.print_end_time()

    # This is a static method to allow for use in coadding script 
    @staticmethod
    def select_detectors(detnum=None, ndet=1):
        """
        Return the 1-indexed list of detectors to reduce.

        Args:
            detnum (:obj:`int`, :obj:`list`, optional):
                One or more detectors to reduce.  If None, return the
                full list for the provided number of detectors (`ndet`).
            ndet (:obj:`int`, optional):
                The number of detectors for this instrument.  Only used
                if `detnum is None`.

        Returns:
            list:  List of detectors to be reduced

        """
        if detnum is None:
            return np.arange(1, ndet+1).tolist()
        else:
            return np.atleast_1d(detnum).tolist()

    def msgs_reset(self):
        """
        Reset the msgs object
        """

        # Reset the global logger
        msgs.reset(log=self.logname, verbosity=self.verbosity)
        msgs.pyphot_file = self.pyphot_file

    def print_end_time(self):
        """
        Print the elapsed time
        """
        # Capture the end time and print it to user
        tend = time.time()
        codetime = tend-self.tstart
        if codetime < 60.0:
            msgs.info('Execution time: {0:.2f}s'.format(codetime))
        elif codetime/60.0 < 60.0:
            mns = int(codetime/60.0)
            scs = codetime - 60.0*mns
            msgs.info('Execution time: {0:d}m {1:.2f}s'.format(mns, scs))
        else:
            hrs = int(codetime/3600.0)
            mns = int(60.0*(codetime/3600.0 - hrs))
            scs = codetime - 60.0*mns - 3600.0*hrs
            msgs.info('Execution time: {0:d}h {1:d}m {2:.2f}s'.format(hrs, mns, scs))

    def __repr__(self):
        # Generate sets string
        return '<{:s}: pyphot_file={}>'.format(self.__class__.__name__, self.pyphot_file)


