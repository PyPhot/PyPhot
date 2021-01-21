"""
Main driver class for PyPhot run

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst

"""
import time
import os
import numpy as np
from astropy import stats
from astropy.io import fits

from configobj import ConfigObj

from pyphot import msgs
from pyphot import procimg
from pyphot import sex, scamp, swarp
from pyphot import query, crossmatch
from pyphot.par.util import parse_pyphot_file
from pyphot.par import PyPhotPar
from pyphot.metadata import PyPhotMetaData
from pyphot.cameras.util import load_camera
from pyphot import masterframe

from IPython import embed

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

        if self.calibrations_path is not None and not os.path.isdir(self.calibrations_path):
            os.makedirs(self.calibrations_path)
        if self.science_path is not None and not os.path.isdir(self.science_path):
            os.makedirs(self.science_path)
        if self.qa_path is not None and not os.path.isdir(self.qa_path):
            os.makedirs(self.qa_path)

        # Report paths
        msgs.info('Setting reduction path to {0}'.format(self.par['rdx']['redux_path']))
        msgs.info('Master calibration data output to: {0}'.format(self.calibrations_path))
        msgs.info('Science data output to: {0}'.format(self.science_path))
        msgs.info('Quality assessment plots output to: {0}'.format(self.qa_path))
        # TODO: Is anything written to the qa dir or only to qa/PNGs?
        # Should we have separate calibration and science QA
        # directories?
        # An html file wrapping them all too

        # Init
        # TODO: I don't think this ever used

        self.det = None

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
        self.par.validate_keys(required=['rdx', 'calibrations', 'scienceframe', 'reduce',
                                         'flexure'])
        self.tstart = time.time()

        # Find the standard frames
        is_standard = self.fitstbl.find_frames('standard')

        # Find the dark frames
        is_dark = self.fitstbl.find_frames('dark')

        # Find the pixel flat frames
        is_pixflat = self.fitstbl.find_frames('pixelflat')

        # Find the illuminate flat frames
        is_illumflat = self.fitstbl.find_frames('illumflat')

        # Find the science frames
        is_science = self.fitstbl.find_frames('science')

        # Frame indices
        frame_indx = np.arange(len(self.fitstbl))

        # Iterate over each calibration group again and reduce the science frames
        for i in range(self.fitstbl.n_calib_groups):
            # Find all the frames in this calibration group
            in_grp = self.fitstbl.find_calib_group(i)

            if np.sum(in_grp)<1:
                msgs.info('No frames found for the {:}th calibration group, skipping.'.format(i))
            else:
                grp_frames = frame_indx[in_grp]
                # Find the indices of the science frames in this calibration group:
                grp_science = frame_indx[is_science & in_grp]

                # Find the detectors to reduce
                detectors = PyPhot.select_detectors(detnum=self.par['rdx']['detnum'],
                                                ndet=self.camera.ndet)

                # Loop on Detectors for calibrations and processing images
                for self.det in detectors:

                    this_setup = np.unique(self.fitstbl.table[grp_science]['setup'])[0]
                    this_det = self.det
                    this_name = '{:}_{:02d}.fits'.format(this_setup,this_det)

                    ### Build Calibrations
                    # Bias
                    if self.par['scienceframe']['process']['use_biasimage']:
                        masterbias_name = os.path.join(self.par['calibrations']['master_dir'], 'MasterBias_{:}'.format(this_name))
                        if os.path.exists(masterbias_name) and self.reuse_masters:
                            msgs.info('Using existing master file {:}'.format(masterbias_name))
                        else:
                            msgs.work('Build Master Bias')
                        masterbiasimg = fits.getdata(masterbias_name,1)
                    else:
                        masterbiasimg = None

                    # Dark
                    if self.par['scienceframe']['process']['use_darkimage']:
                        masterdark_name = os.path.join(self.par['calibrations']['master_dir'], 'MasterDark_{:}'.format(this_name))

                        if os.path.exists(masterdark_name) and self.reuse_masters:
                            msgs.info('Using existing master file {:}'.format(masterdark_name))
                        else:
                            grp_dark = frame_indx[is_dark & in_grp]
                            darkfiles = self.fitstbl.frame_paths(grp_dark)
                            masterframe.darkframe(darkfiles, self.camera, self.det, masterdark_name, masterbiasimg=masterbiasimg,
                                                  cenfunc='median', stdfunc='std', sigma=3, maxiters=5)
                        masterdarkimg = fits.getdata(masterdark_name, 1)
                    else:
                        masterdarkimg = None

                    # Pixel Flat
                    if self.par['scienceframe']['process']['use_pixelflat']:
                        masterpixflat_name = os.path.join(self.par['calibrations']['master_dir'], 'MasterPixelFlat_{:}'.format(this_name))

                        if os.path.exists(masterpixflat_name) and self.reuse_masters:
                            msgs.info('Using existing master file {:}'.format(masterpixflat_name))
                        else:
                            grp_pixflat = frame_indx[is_pixflat & in_grp]
                            pixflatfiles = self.fitstbl.frame_paths(grp_pixflat)
                            masterframe.pixelflatframe(pixflatfiles, self.camera, self.det, masterpixflat_name,
                                                  masterbiasimg=masterbiasimg, masterdarkimg=masterdarkimg,
                                                  cenfunc='median', stdfunc='std', sigma=3, maxiters=5)
                        masterpixflatimg = fits.getdata(masterpixflat_name, 1)
                    else:
                        masterpixflatimg = None

                    # Illumination Flat
                    if self.par['scienceframe']['process']['use_illumflat']:
                        masterillumflat_name = os.path.join(self.par['calibrations']['master_dir'],
                                                          'MasterIllumFlat_{:}'.format(this_name))

                        if os.path.exists(masterillumflat_name) and self.reuse_masters:
                            msgs.info('Using existing master file {:}'.format(masterillumflat_name))
                        else:
                            grp_illumflat = frame_indx[is_illumflat & in_grp]
                            illumflatfiles = self.fitstbl.frame_paths(grp_illumflat)
                            masterframe.illumflatframe(illumflatfiles, self.camera, self.det, masterillumflat_name,
                                                       masterbiasimg=masterbiasimg, masterdarkimg=masterdarkimg,
                                                       masterpixflatimg=masterpixflatimg,
                                                       cenfunc='median', stdfunc='std', sigma=3, maxiters=5)
                        masterillumflatimg = fits.getdata(masterillumflat_name, 1)
                    else:
                        masterillumflatimg = None

                    ## Procimage
                    scifiles = self.fitstbl.frame_paths(grp_science)
                    sci_fits_list, wht_fits_list, flag_fits_list = procimg.sciproc(scifiles, self.camera, self.det,
                                    science_path=self.science_path,masterbiasimg=masterbiasimg, masterdarkimg=masterdarkimg,
                                    masterpixflatimg=masterpixflatimg, masterillumflatimg=masterillumflatimg,
                                    background='median', boxsize=(50, 50), filter_size=(3, 3), #ToDo: add to parset
                                    maxiter=self.par['scienceframe']['process']['lamaxiter'],
                                    grow=self.par['scienceframe']['process']['grow'],
                                    remove_compact_obj=self.par['scienceframe']['process']['rmcompact'],
                                    sigclip=self.par['scienceframe']['process']['sigclip'],
                                    sigfrac=self.par['scienceframe']['process']['sigfrac'],
                                    objlim=self.par['scienceframe']['process']['objlim'])

                    quicklook=False
                    detector_par = self.camera.get_detector_par(fits.open(scifiles[0]), self.det)
                    pixscale = detector_par['det{:02d}'.format(self.det)]['platescale']

                    if quicklook:
                        msgs.warn('This is quick look, skipping individual image calibrations. Go with luck')
                    else:
                        ## Calibrate images
                        ## ToDo: Add parset for the following parameters
                        ## ToDo: Not sure why, but the scamp fails for MMIRS data, so I will try to resample it first and then do the following
                        # configuration for the first swarp run
                        swarpconfig0 ={"RESAMPLE": "Y", "DELETE_TMPFILES": "Y", "CENTER_TYPE": "ALL", "PIXELSCALE_TYPE": "MANUAL",
                                       "PIXEL_SCALE": pixscale, "SUBTRACT_BACK": "N","COMBINE_TYPE": "MEDIAN",
                                       "RESAMPLE_SUFFIX": ".resamp.fits", "WEIGHT_TYPE": "NONE"}
                        # resample science image
                        swarp.swarpall(sci_fits_list, config=swarpconfig0, workdir=self.science_path, defaultconfig='pyphot',
                                       coaddroot=None, delete=False, log=True)
                        # resample weight image
                        #swarp.swarpall(wht_fits_list, config=swarpconfig0, workdir=self.science_path, defaultconfig='pyphot',
                        #               coaddroot=None, delete=False, log=True)
                        # resample flag image
                        #swarp.swarpall(flag_fits_list, config=swarpconfig0, workdir=self.science_path, defaultconfig='pyphot',
                        #               coaddroot=None, delete=False, log=True)

                        ## remove useless data
                        sci_fits_list_resample = []
                        wht_fits_list_resample = []
                        flag_fits_list_resample = []
                        ## rename the resampled images
                        for i in range(len(sci_fits_list)):
                            os.system('rm {:}'.format(sci_fits_list[i].replace('.fits','.0001.resamp.weight.fits')))
                            #os.system('rm {:}'.format(wht_fits_list[i].replace('.fits','.0001.resamp.weight.fits')))
                            #os.system('rm {:}'.format(flag_fits_list[i].replace('.fits','.0001.resamp.weight.fits')))
                            os.system('mv {:} {:}'.format(sci_fits_list[i].replace('.fits','.0001.resamp.fits'),
                                                          sci_fits_list[i].replace('.fits','.resamp.fits')))
                            #os.system('mv {:} {:}'.format(wht_fits_list[i].replace('.fits','.0001.resamp.fits'),
                            #                              wht_fits_list[i].replace('.fits','.resamp.fits')))
                            #os.system('mv {:} {:}'.format(flag_fits_list[i].replace('.fits','.0001.resamp.fits'),
                            #                              flag_fits_list[i].replace('.fits','.resamp.fits')))
                            sci_fits_list_resample.append(sci_fits_list[i].replace('.fits','.resamp.fits'))
                            wht_fits_list_resample.append(wht_fits_list[i].replace('.fits', '.resamp.fits'))
                            flag_fits_list_resample.append(flag_fits_list[i].replace('.fits', '.resamp.fits'))

                        # configuration for the first SExtractor run
                        #embed()

                        sexconfig0 = {"CHECKIMAGE_TYPE": "NONE", "WEIGHT_TYPE": "NONE", "CATALOG_NAME": "dummy.cat",
                                      "CATALOG_TYPE": "FITS_LDAC", "DETECT_THRESH":3.0, "ANALYSIS_THRESH":3.0, "DETECT_MINAREA":5}
                        sexparams0 = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE',
                                     'ERRTHETAWIN_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'ISOAREAF_IMAGE', 'ISOAREA_IMAGE','ELLIPTICITY',
                                     'ELONGATION', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_APER','MAGERR_APER']
                        sex.sexall(sci_fits_list_resample, task='sex', config=sexconfig0, workdir=self.science_path, params=sexparams0,
                                   defaultconfig='pyphot', conv='995', nnw=None, dual=False, delete=False, log=True)

                        # configuration for the first scamp run
                        #
                        scampconfig0 = {"CROSSID_RADIUS": 2.0, "ASTREF_CATALOG": "GAIA-DR2", "ASTREF_BAND": "DEFAULT",
                                        "POSITION_MAXERR": 0.5, "PIXSCALE_MAXERR":1.1, "MOSAIC_TYPE": "UNCHANGED"}
                        #scampconfig0 = {"POSITION_MAXERR": 1.0, "POSANGLE_MAXERR": 10.0, "CROSSID_RADIUS": 2.0,
                        #                "MOSAIC_TYPE": "LOOSE", "SOLVE_PHOTOM": "N",
                        #                "MAGZERO_OUT": 0.0, "CHECKPLOT_TYPE": "None", "ASTREF_CATALOG": "GAIA-DR2", "ASTREF_BAND": "DEFAULT"}
                        scamp.scampall(sci_fits_list_resample, config=scampconfig0, workdir=self.science_path, defaultconfig='pyphot',
                                       delete=False, log=True)

                        # configuration for the first swarp run
                        swarpconfig0 ={"RESAMPLE": "Y", "DELETE_TMPFILES": "Y", "CENTER_TYPE": "ALL", "PIXELSCALE_TYPE": "MANUAL",
                                       "PIXEL_SCALE": pixscale, "SUBTRACT_BACK": "N","COMBINE_TYPE": "MEDIAN",
                                       "RESAMPLE_SUFFIX": ".astrometry.fits", "WEIGHT_TYPE": "NONE"}
                        swarp.swarpall(sci_fits_list_resample, config=swarpconfig0, workdir=self.science_path, defaultconfig='pyphot',
                                       coaddroot=None, delete=False, log=True)

                ## Coadd images target by target
                coadddir = os.path.join(self.science_path,'Coadd')
                #if not os.path.exists(coadddir):
                #    os.makedirs(coadddir)
                ## Combine images using combid
                aper = 3.0 # in units of arcsec, ToDo: parset
                ndet = np.size(detectors)
                objids = np.unique(self.fitstbl['comb_id'][grp_science]) ## number of combine groups
                for objid in objids:
                    grp_iobj = frame_indx[is_science & in_grp & (self.fitstbl['comb_id']==objid)]
                    iobjfiles = self.fitstbl['filename'][grp_iobj]
                    filter_iobj = self.fitstbl['filter'][grp_iobj][0]
                    coaddroot = self.fitstbl['target'][grp_iobj][0]+'_coadd_combid{:03d}'.format(objid)
                    nscifits = np.size(iobjfiles)
                    #calfiles = np.repeat(iobjfiles.data,ndet)
                    scifiles_iobj= []
                    for idet in detectors:
                        for ii in range(nscifits):
                            iname = os.path.join(self.science_path,iobjfiles[ii].replace('.fits',
                                                 '_det{:02d}_sci.resamp.astrometry.fits'.format(idet)))
                            scifiles_iobj.append(iname)

                    # configuration for the swarp run
                    # ToDo: need flag image
                    msgs.info('Coadding image for {:}'.format(coaddroot))
                    swarpconfig = {"RESAMPLE": "Y", "DELETE_TMPFILES": "Y", "CENTER_TYPE": "ALL",
                                    "PIXELSCALE_TYPE": "MANUAL",
                                    "PIXEL_SCALE": pixscale, "SUBTRACT_BACK": "N", "COMBINE_TYPE": "MEDIAN",
                                    "RESAMPLE_SUFFIX": ".resamp.fits", "WEIGHT_TYPE": "NONE"}
                    swarp.swarpall(scifiles_iobj, config=swarpconfig, workdir=self.science_path, defaultconfig='pyphot',
                                   coaddroot=coaddroot, delete=True, log=False)

                    # configuration for the sextractor run
                    sexconfig = {"CHECKIMAGE_TYPE": "NONE", "WEIGHT_TYPE": "NONE", "CATALOG_NAME": "dummy.cat",
                                  "CATALOG_TYPE": "FITS_LDAC", "DETECT_THRESH": 2.0, "ANALYSIS_THRESH": 2.0,
                                  "DETECT_MINAREA": 3, "PHOT_APERTURES": aper / pixscale}
                    sex.sexone(os.path.join(coadddir,coaddroot+'.fits'), task='sex', config=sexconfig, workdir=coadddir, params=None,
                               defaultconfig='pyphot', conv='995', nnw=None, dual=False, delete=True, log=False)

                    # refine the astrometry with the coadded image against with GAIA
                    scampconfig = {"CROSSID_RADIUS": 2.0, "ASTREF_CATALOG": "GAIA-DR2", "ASTREF_BAND": "DEFAULT",
                                    "PIXSCALE_MAXERR": 1.1, "MOSAIC_TYPE": "UNCHANGED"}
                    scamp.scampone(os.path.join(coadddir,coaddroot+'.fits'), config=scampconfig, workdir=coadddir, defaultconfig='pyphot',
                                   delete=False, log=True)
                    swarp.swarpone(os.path.join(coadddir,coaddroot+'.fits'), config=swarpconfig, workdir=coadddir, defaultconfig='pyphot',
                                   delete=True, log=False)
                    sex.sexone(os.path.join(coadddir,coaddroot+'.resamp.fits'), task='sex', config=sexconfig, workdir=coadddir, params=None,
                               defaultconfig='pyphot', conv='995', nnw=None, dual=False, delete=True, log=False)

                    # calibrate it against with 2MASS
                    sextable = fits.getdata(os.path.join(coadddir,coaddroot+'.resamp.cat'), 2)
                    rasex, decsex= sextable['ALPHA_J2000'], sextable['DELTA_J2000']
                    magsex, magerrsex = sextable['MAG_AUTO'],sextable['MAGERR_AUTO']
                    possex = np.zeros((len(rasex), 2))
                    possex[:, 0], possex[:, 1] = rasex, decsex

                    result_2mass = query.twomass(rasex[0], decsex[0], radius='1deg')
                    ra2mass, dec2mass = result_2mass['RAJ2000'], result_2mass['DEJ2000']
                    jmag, jmagerr = result_2mass['Jmag'], result_2mass['e_Jmag']
                    pos2mass = np.zeros((len(ra2mass), 2))
                    pos2mass[:, 0], pos2mass[:, 1] = ra2mass, dec2mass

                    dist, ind = crossmatch.crossmatch_angular(possex, pos2mass, max_distance=3.0 / 3600.)
                    dist_good = dist[np.invert(np.isinf(dist))]
                    dist_mean, dist_median, dist_std = stats.sigma_clipped_stats(dist_good,sigma=3, maxiters=20,cenfunc='median', stdfunc='std')
                    matched = np.invert(np.isinf(dist)) & (dist>dist_median-dist_std) & (dist<dist_median+dist_std)

                    sextable = sextable[matched]
                    result_2mass = result_2mass[ind[matched]]
                    nstar = len(ind[matched])

                    _, zp, zp_std = stats.sigma_clipped_stats(result_2mass["{:}mag".format(filter_iobj)] - sextable['MAG_AUTO'],
                                                              sigma=3, maxiters=20,cenfunc='median', stdfunc='std')
                    # rerun the SExtractor with the zero point
                    msgs.warn('Zeropoint is {:}+/-{:}'.format(zp,zp_std))
                    sexconfig = {"CHECKIMAGE_TYPE": "NONE", "WEIGHT_TYPE": "NONE", "CATALOG_NAME": "dummy.cat",
                                 "CATALOG_TYPE": "FITS_LDAC", "DETECT_THRESH": 2.0, "ANALYSIS_THRESH": 2.0,
                                 "DETECT_MINAREA": 3, "PHOT_APERTURES": aper / pixscale, "MAG_ZEROPOINT": zp}
                    sex.sexone(os.path.join(coadddir, coaddroot + '.resamp.fits'), task='sex', config=sexconfig, workdir=coadddir,
                               params=None, defaultconfig='pyphot', conv='995', nnw=None, dual=False, delete=True, log=False)

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


