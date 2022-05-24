"""
Main driver class for PyPhot run

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst

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
        show: (:obj:`bool`, optional):
            Show reduction steps via plots (which will block further
            execution until clicked on) and outputs to ginga. Requires
            remote control ginga session via ``ginga --modules=RC &``
        redux_path (:obj:`str`, optional):
            Over-ride reduction path in PyPhot file (e.g. Notebook usage)
        calib_only: (:obj:`bool`, optional):
            Only generate the calibration files that you can

    Attributes:
        pyphot_file (:obj:`str`):
            Name of the pyphot file to read.  PyPhot files have a
            specific set of valid formats. A description can be found
            :ref:`pyphot_file`.
        fitstbl (:obj:`pyphot.metadata.PyPhotMetaData`): holds the meta info

    """

    def __init__(self, pyphot_file, verbosity=2, overwrite=True, reuse_masters=False, logname=None,
                 show=False, redux_path=None, calib_only=False):

        # Set up logging
        self.logname = logname
        self.verbosity = verbosity
        self.pyphot_file = pyphot_file
        
        self.msgs_reset()

        # Load
        cfg_lines, data_files, frametype, usrdata, setups \
                = parse_pyphot_file(pyphot_file, runtime=True)
        self.calib_only = calib_only

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
        self.show = show

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

    def build_qa(self):
        """
        Generate QA wrappers
        """
        msgs.work("TBD")

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

        ## Step one: process and calibrate individual exposures, do it chips by chips
        # It includes: bias, dark subtraction, flat fielding, super-sky flattening, Fringe subtraction
        #              cosmic ray rejection, astrometry calibration, and photometric calibrations
        #              all these steps are performed chips by chips.
        # Iterate over each calibration group and reduce the science frames
        if self.par['rdx']['skip_step_one']:
            msgs.warn('Skipping all the calibrations and individual chip processing')
        else:
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

                    scifiles = self.fitstbl.frame_paths(grp_science)  # list for scifiles
                    sci_airmass = self.fitstbl[grp_science]['airmass']

                    procfiles = self.fitstbl.frame_paths(grp_proc)  # need run detproc
                    proc_airmass = self.fitstbl[grp_proc]['airmass']

                    superskyfiles = self.fitstbl.frame_paths(grp_supersky) # supersky files
                    fringefiles = self.fitstbl.frame_paths(in_grp_fringe) # Fringe files

                    sciprocfiles = self.fitstbl.frame_paths(grp_sciproc)  # need run both detproc and sciproc
                    sciproc_airmass = self.fitstbl[grp_sciproc]['airmass']

                    coadd_ids = self.fitstbl['coadd_id'][grp_science] # coadd_ids
                    sci_ra = self.fitstbl['ra'][grp_science]
                    sci_dec = self.fitstbl['dec'][grp_science]

                    # Loop on Detectors for calibrations
                    masterbiasimg_list = []
                    masterdarkimg_list = []
                    masterillumflatimg_list = []
                    masterpixflatimg_list = []
                    bpm_proc_list = []
                    norm_illum_list = []
                    norm_pixel_list = []

                    ## Build MasterFrames, including bias, dark, illumflat, and pixelflat
                    for idet in detectors:
                        master_key = self.fitstbl.master_key(grp_all[0], det=idet)
                        msgs.info('Identify data size for detector {:} based on BPM image.'.format(idet))
                        raw_shape = self.camera.bpm(allfiles[0], idet, shape=None, msbias=None).astype('bool').shape
                        Master = masterframe.MasterFrames(self.par, self.camera, idet, master_key, raw_shape,
                                                          reuse_masters=self.reuse_masters)

                        # Build MasterFrames
                        if not self.par['rdx']['skip_master']:
                            grp_bias = frame_indx[is_bias & in_grp]
                            biasfiles = self.fitstbl.frame_paths(grp_bias)

                            grp_dark = frame_indx[is_dark & in_grp]
                            darkfiles = self.fitstbl.frame_paths(grp_dark)

                            grp_illumflat = frame_indx[is_illumflat & in_grp]
                            illumflatfiles = self.fitstbl.frame_paths(grp_illumflat)

                            grp_pixflat = frame_indx[is_pixflat & in_grp]
                            pixflatfiles = self.fitstbl.frame_paths(grp_pixflat)

                            Master.build(biasfiles=biasfiles, darkfiles=darkfiles,
                                         illumflatfiles=illumflatfiles, pixflatfiles=pixflatfiles)

                        # Load master frames
                        masterbiasimg, masterdarkimg, masterillumflatimg, masterpixflatimg, bpm_proc,\
                            norm_illum, norm_pixel = Master.load()
                        masterbiasimg_list.append(masterbiasimg)
                        masterdarkimg_list.append(masterdarkimg)
                        masterillumflatimg_list.append(masterillumflatimg)
                        masterpixflatimg_list.append(masterpixflatimg)
                        bpm_proc_list.append(bpm_proc)
                        norm_illum_list.append(norm_illum)
                        norm_pixel_list.append(norm_pixel)

                    ## ToDo: Re-scale the Flat normalizations in different detectors?
                    #if not skip_build_master:
                    #    scale_illum = norm_illum_list / np.median(norm_illum_list)
                    #    scale_pixel = norm_pixel_list / np.median(norm_pixel_list)
                    #    #if self.par['scienceframe']['process']['use_illumflat']:
                    #    if self.par['scienceframe']['process']['use_pixelflat']:
                    #        for ii, idet in enumerate(detectors):
                    #            master_key = self.fitstbl.master_key(grp_science[0], det=idet)
                    #            masterpixflat_name = os.path.join(self.par['calibrations']['master_dir'],
                    #                                              'MasterPixelFlat_{:}'.format(master_key))
                    #            headerpixel, masterpixflatimg, maskpixflatimg = io.load_fits(masterpixflat_name)
                    #            headerpixel['FScale'] = scale_pixel[ii]
                    #            io.save_fits(masterpixflat_name, masterpixflatimg*scale_pixel[ii], headerpixel,
                    #                         'MasterPixelFlat', mask=maskpixflatimg, overwrite=True)

                if np.sum(in_grp_sci) > 0:
                    ## Data processing, including detproc, supersky, sciproc, fringing
                    ## Loop over detectors for sciproc
                    for ii, idet in enumerate(detectors):
                        master_key = self.fitstbl.master_key(grp_science[0], det=idet)
                        raw_shape = self.camera.bpm(allfiles[0], idet, shape=None, msbias=None).astype('bool').shape
                        ## Initialize ImageProc
                        Proc = procimg.ImageProc(self.par, self.camera, idet, self.science_path, master_key, raw_shape,
                                                 reuse_masters=self.reuse_masters)
                        ## DETPROC -- bias, dark subtraction and flat fielding, support parallel processing
                        if not self.par['rdx']['skip_detproc']:
                            Proc.run_detproc(procfiles, masterbiasimg_list[ii], masterdarkimg_list[ii],
                                             masterpixflatimg_list[ii], masterillumflatimg_list[ii], bpm_proc_list[ii])

                        ## SCIPROC -- supersky flattening, extinction correction based on airmass, and background subtraction.
                        if not self.par['rdx']['skip_sciproc']:

                            ## Build SuperSkyFlat first
                            if self.par['scienceframe']['process']['use_supersky']:
                                Proc.build_supersky(superskyfiles)

                            ## Run sciproc
                            Proc.run_sciproc(sciprocfiles, sciproc_airmass)

                            ## Build Master Fringing and Defringing
                            # ToDo: should I move defringing into sciproc?
                            if self.par['scienceframe']['process']['use_fringe']:
                                Proc.build_fringe(fringefiles)
                                # Defringing
                                Proc.run_defringing(scifiles)

                        ## The following would be useless if the astrometry would be done globaly.
                        sci_fits_list, wht_fits_list, flag_fits_list = [], [], []
                        for ifile in scifiles:
                            rootname = os.path.join(self.science_path, os.path.basename(ifile))
                            sci_fits_file = rootname.replace('.fits', '_det{:02d}_sci.fits'.format(idet))
                            wht_fits_file = rootname.replace('.fits', '_det{:02d}_sci.weight.fits'.format(idet))
                            flag_fits_file = rootname.replace('.fits', '_det{:02d}_flag.fits'.format(idet))
                            sci_fits_list.append(sci_fits_file)
                            wht_fits_list.append(wht_fits_file)
                            flag_fits_list.append(flag_fits_file)

                    ## Astrometry
                    photref_catalog = self.par['postproc']['photometry']['photref_catalog']
                    # pixel scale
                    pixscales = []
                    for idet in detectors:
                        detector_par = self.camera.get_detector_par(fits.open(scifiles[0]), idet)
                        pixscales.append(detector_par['platescale'])
                    pixscale = np.median(pixscales)

                    # Initiallize
                    Astrometry = postproc.Astrometry(self.par, detectors, this_setup, scifiles, coadd_ids, sci_ra, sci_dec,
                                                     pixscale, self.science_path, self.qa_path, reuse_masters=self.reuse_masters)
                    if not self.par['rdx']['skip_astrometry']:
                        # run it
                        Astrometry.run_astrometry()

                    ## Photometrically calibrating individual chips
                    if self.par['postproc']['photometry']['cal_chip_zpt']:
                        msgs.info('Photometrically calibrating individual chips.')
                        # Do the calibrations
                        # ToDo: It seems that set n_process>1 could cause problem for downloading reference catalogs in some cases.
                        zp_all, zp_std_all, nstar_all, fwhm_all = postproc.cal_chips(Astrometry.cat_resample_list,
                                                sci_fits_list=Astrometry.sci_resample_list,
                                                ref_fits_list=Astrometry.master_ref_cats,
                                                outqa_root_list=Astrometry.outqa_list,
                                                refcatalog=self.par['postproc']['photometry']['photref_catalog'],
                                                primary=self.par['postproc']['photometry']['primary'],
                                                secondary=self.par['postproc']['photometry']['secondary'],
                                                coefficients=self.par['postproc']['photometry']['coefficients'],
                                                ZP=self.par['postproc']['photometry']['zpt'],
                                                nstar_min=self.par['postproc']['photometry']['nstar_min'],
                                                external_flag=self.par['postproc']['photometry']['external_flag'],
                                                pixscale=pixscale, n_process=self.par['rdx']['n_process'])

                        # The FITS table that stores individual zero-points
                        master_zpt_name = os.path.join(self.par['calibrations']['master_dir'],
                                            'MasterZPT_{:}_{:}.fits'.format(this_setup,self.fitstbl['filter'][grp_science][0]))
                        #master_zpt_name = os.path.join(self.par['calibrations']['master_dir'],'MasterZPT_{:}'.format(master_key))
                        master_zpt_tbl = Table()
                        master_zpt_tbl['Name'] = [os.path.basename(i) for i in Astrometry.sci_resample_list]
                        #master_zpt_tbl['Name'] = self.fitstbl['filename'][grp_science].astype('U20')
                        #master_zpt_tbl['filter'] = self.fitstbl['filter'][grp_science].astype('U10')
                        master_zpt_tbl['exptime'] = np.repeat(self.fitstbl['exptime'][grp_science].astype('double'),len(detectors))
                        master_zpt_tbl['airmass'] = np.repeat(self.fitstbl['airmass'][grp_science].astype('double'),len(detectors))
                        master_zpt_tbl['ZPT'] = zp_all
                        master_zpt_tbl['ZPT_Std'] = zp_std_all
                        master_zpt_tbl['FWHM'] = fwhm_all
                        master_zpt_tbl['NStar'] = nstar_all.astype('int32')
                        #master_zpt_tbl['Detector'] = (np.ones_like(nstar_all) * idet).astype('int32')
                        master_zpt_tbl.write(master_zpt_name, overwrite=True)

                    ## Making QA image for calibrated individual chips
                    if not self.par['rdx']['skip_img_qa']:
                        outroots = []
                        for this_image in Astrometry.sci_resample_list:
                            outroots.append(os.path.join(self.qa_path, os.path.basename(this_image).replace('.fits','_img')))
                        utils.showimages(Astrometry.sci_resample_list, outroots=outroots,
                                         interval_method=self.par['postproc']['qa']['interval_method'],
                                         vmin=self.par['postproc']['qa']['vmin'],
                                         vmax=self.par['postproc']['qa']['vmax'],
                                         stretch_method=self.par['postproc']['qa']['stretch_method'],
                                         cmap=self.par['postproc']['qa']['cmap'],
                                         plot_wcs=self.par['postproc']['qa']['plot_wcs'],
                                         show=self.par['postproc']['qa']['show'],
                                         n_process=self.par['rdx']['n_process'])

                    ## ToDo: combine different detectors for each exposure. Do I need to calibrate the zeropoint again here? Probably not?
                    ##       using swarp to combine different detectors, if only one detector then skip this step.
                    ##       RESAMPLING_TYPE = NEAREST,
                    ##       Not sure whether its usful or not given that we can just ds9 -mosaic **resample.fits to check the image.

        ## Step two: coadding, second pass on the photometric calibration, and source detection
        ## The images are combined based on the coadd_id in your PyPhot file.
        ## Make sure to use different coadd_id for different filters
        if self.par['rdx']['skip_step_two']:
            msgs.warn('Skipping all the coadding, detection and photometry')
        else:
            objids = np.unique(self.fitstbl['coadd_id'][is_science]) ## number of combine groups
            for objid in objids:
                grp_iobj = frame_indx[is_science & (self.fitstbl['coadd_id']==objid)] #& in_grp
                iobjfiles = self.fitstbl['filename'][grp_iobj]
                filter_iobj = self.fitstbl['filter'][grp_iobj][0]
                coaddroot = self.fitstbl['target'][grp_iobj][0]+'_{:}_coadd_ID{:03d}'.format(filter_iobj,objid)
                if ('.gz' in iobjfiles[0]) or ('.fz' in iobjfiles[0]):
                    for ii in range(len(iobjfiles)):
                        iobjfiles[ii] = iobjfiles[ii].replace('.gz','').replace('.fz','')
                # The name of reference catalog that will be saved to Master folder
                out_refcat = 'MasterRefCat_{:}_ID{:03d}.fits'.format(self.par['postproc']['photometry']['photref_catalog'],objid)
                out_refcat_fullpath = os.path.join(self.par['calibrations']['master_dir'], out_refcat)
                # pixscale
                if self.par['postproc']['coadd']['pixscale'] is None:
                    # get pixel scale for resampling with SCAMP
                    detector_par = self.camera.get_detector_par(fits.open(self.fitstbl.frame_paths(grp_iobj)[0]), 1)
                    pixscale = detector_par['platescale']
                else:
                    pixscale = self.par['postproc']['coadd']['pixscale']

                # compile the file list
                nscifits = np.size(iobjfiles)
                scifiles_iobj= []
                flagfiles_iobj= []
                whtfiles_iobj= []
                for idet in detectors:
                    for ii in range(nscifits):
                        #if self.par['postproc']['astrometry']['skip_astrometry']:
                        if os.path.exists(os.path.join(self.science_path,iobjfiles[ii].replace('.fits',
                                                       '_det{:02d}_sci.resamp.fits'.format(idet)))):
                            this_sci = os.path.join(self.science_path,iobjfiles[ii].replace('.fits',
                                                 '_det{:02d}_sci.resamp.fits'.format(idet)))
                            this_flag = os.path.join(self.science_path,iobjfiles[ii].replace('.fits',
                                                 '_det{:02d}_flag.resamp.fits'.format(idet)))
                            this_wht = os.path.join(self.science_path,iobjfiles[ii].replace('.fits',
                                                 '_det{:02d}_sci.resamp.weight.fits'.format(idet)))
                        else:
                            this_sci = os.path.join(self.science_path,iobjfiles[ii].replace('.fits',
                                                 '_det{:02d}_sci.fits'.format(idet)))
                            this_flag = os.path.join(self.science_path,iobjfiles[ii].replace('.fits',
                                                 '_det{:02d}_flag.fits'.format(idet)))
                            this_wht = os.path.join(self.science_path,iobjfiles[ii].replace('.fits',
                                                 '_det{:02d}_sci.weight.fits'.format(idet)))

                        scifiles_iobj.append(this_sci)
                        flagfiles_iobj.append(this_flag)
                        whtfiles_iobj.append(this_wht)

                ## Do it
                if self.par['rdx']['skip_coadd']:
                    msgs.warn('Skipping coadding process. Make sure you have produced the coadded images !!!')
                else:
                    coadd_file, coadd_wht_file, coadd_flag_file = postproc.coadd(scifiles_iobj, flagfiles_iobj, coaddroot,
                                                pixscale, self.science_path, self.coadd_path,
                                                weight_type=self.par['postproc']['coadd']['weight_type'],
                                                rescale_weights=self.par['postproc']['coadd']['rescale_weights'],
                                                combine_type=self.par['postproc']['coadd']['combine_type'],
                                                clip_ampfrac=self.par['postproc']['coadd']['clip_ampfrac'],
                                                clip_sigma=self.par['postproc']['coadd']['clip_sigma'],
                                                blank_badpixels=self.par['postproc']['coadd']['blank_badpixels'],
                                                subtract_back=self.par['postproc']['coadd']['subtract_back'],
                                                back_type=self.par['postproc']['coadd']['back_type'],
                                                back_default=self.par['postproc']['coadd']['back_default'],
                                                back_size=self.par['postproc']['coadd']['back_size'],
                                                back_filtersize=self.par['postproc']['coadd']['back_filtersize'],
                                                back_filtthresh=self.par['postproc']['coadd']['back_filtthresh'],
                                                resampling_type=self.par['postproc']['coadd']['resampling_type'],
                                                sextractor_task=self.par['rdx']['sextractor'],
                                                detect_thresh=self.par['postproc']['detection']['detect_thresh'],
                                                analysis_thresh=self.par['postproc']['detection']['analysis_thresh'],
                                                detect_minarea=self.par['postproc']['detection']['detect_minarea'],
                                                delete=self.par['postproc']['coadd']['delete'],
                                                log=self.par['postproc']['coadd']['log'])

                ## calibrate the zeropoint for the final stacked image
                if self.par['postproc']['photometry']['cal_zpt']:
                    msgs.info('Calcuating the zeropoint for {:}'.format(os.path.join(self.coadd_path, coaddroot + '_sci_zptcat.fits')))
                    zpt, zpt_std, nstar, matched_table = postproc.calzpt(os.path.join(self.coadd_path, coaddroot + '_sci_zptcat.fits'),
                                                        refcatalog=self.par['postproc']['photometry']['photref_catalog'],
                                                        primary=self.par['postproc']['photometry']['primary'],
                                                        secondary=self.par['postproc']['photometry']['secondary'],
                                                        coefficients=self.par['postproc']['photometry']['coefficients'],
                                                        FLXSCALE=1.0, FLASCALE=1.0,out_refcat=out_refcat_fullpath,
                                                        external_flag=self.par['postproc']['photometry']['external_flag'],
                                                        nstar_min=self.par['postproc']['photometry']['nstar_min'],
                                                        outqaroot=os.path.join(self.qa_path, coaddroot))

                    if matched_table is not None:
                        star_table = Table()
                        star_table['x'] = matched_table['XWIN_IMAGE']
                        star_table['y'] = matched_table['YWIN_IMAGE']
                        fwhm, _, _, _ = psf.buildPSF(star_table, os.path.join(self.coadd_path, coaddroot + '_sci.fits'), pixscale=pixscale,
                                               outroot=os.path.join(self.qa_path, coaddroot))
                    else:
                        fwhm = 0.
                    par = fits.open(os.path.join(self.coadd_path, coaddroot + '_sci.fits'))
                    par[0].header['ZP'] = (zpt, 'Zero point measured from stars')
                    par[0].header['ZP_STD'] = (zpt_std, 'The standard deviration of ZP')
                    par[0].header['ZP_NSTAR'] = (nstar, 'The number of stars used for ZP and FWHM')
                    par[0].header['FWHM'] = (fwhm, 'FWHM in units of arcsec measured from stars')
                    par.writeto(os.path.join(self.coadd_path, coaddroot + '_sci.fits'),overwrite=True)
                else:
                    zpt = self.par['postproc']['photometry']['zpt']

                ## Detection
                if self.par['postproc']['detection']['skip']:
                    msgs.warn('Skipping detecting process. Make sure you have extracted source catalog !!!')
                else:
                    phot_table, rmsmap, bkgmap = postproc.detect('{:}_sci.fits'.format(coaddroot), outroot=coaddroot,
                                                 flag_image='{:}_flag.fits'.format(coaddroot),
                                                 weight_image='{:}_sci.weight.fits'.format(coaddroot),
                                                 bkg_image=None, rms_image=None, workdir=self.coadd_path,
                                                 detection_method=self.par['postproc']['detection']['detection_method'],
                                                 zpt=zpt, effective_gain=None, pixscale=pixscale,
                                                 detect_thresh=self.par['postproc']['detection']['detect_thresh'],
                                                 analysis_thresh=self.par['postproc']['detection']['analysis_thresh'],
                                                 detect_minarea=self.par['postproc']['detection']['detect_minarea'],
                                                 fwhm=self.par['postproc']['detection']['fwhm'],
                                                 nlevels=self.par['postproc']['detection']['nlevels'],
                                                 contrast=self.par['postproc']['detection']['contrast'],
                                                 back_type=self.par['postproc']['detection']['back_type'],
                                                 back_rms_type=self.par['postproc']['detection']['back_rms_type'],
                                                 back_size=self.par['postproc']['detection']['back_size'],
                                                 back_filter_size=self.par['postproc']['detection']['back_filtersize'],
                                                 back_default=self.par['postproc']['detection']['back_default'],
                                                 backphoto_type=self.par['postproc']['detection']['backphoto_type'],
                                                 backphoto_thick=self.par['postproc']['detection']['backphoto_thick'],
                                                 weight_type=self.par['postproc']['detection']['weight_type'],
                                                 check_type=self.par['postproc']['detection']['check_type'],
                                                 back_nsigma=self.par['postproc']['detection']['back_nsigma'],
                                                 back_maxiters=self.par['postproc']['detection']['back_maxiters'],
                                                 morp_filter=self.par['postproc']['detection']['morp_filter'],
                                                 defaultconfig='pyphot', dual=False,
                                                 conv=self.par['postproc']['detection']['conv'],
                                                 nnw=self.par['postproc']['detection']['nnw'],
                                                 delete=self.par['postproc']['detection']['delete'],
                                                 log=self.par['postproc']['detection']['log'],
                                                 sextractor_task=self.par['rdx']['sextractor'],
                                                 phot_apertures=self.par['postproc']['detection']['phot_apertures'])
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


