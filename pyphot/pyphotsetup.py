"""
Class for organizing PyPhot setup

Bookkeeping from PyPeIt.
"""
import os
import inspect
import datetime
import numpy as np

from astropy.table import Table

from pyphot import msgs
from pyphot.metadata import PyPhotMetaData

from pyphot.par import PyPhotPar
from pyphot.par.util import parse_pyphot_file, make_pyphot_file
from pyphot.cameras.util import load_camera


class PyPhotSetup:
    """
    Prepare for a pyphot run.

    .. todo::
        - This is now mostly a wrapper for PyPhotMetaData.  Should we
          remove this class, or merge PyPhotSetup and PyPhotMetaData.

    The main deliverables are the set of parameters used for pyphot's
    algorithms (:attr:`par`), an :obj:`astropy.table.Table` with the
    details of the files to be reduced (:attr:`fitstbl`), and a
    dictionary with the list of instrument setups.

    Args:
        file_list (list):
            A list of strings with the full path to each file to be
            reduced.
        frametype (:obj:`dict`, optional):
            A dictionary that associates the name of the file (just the
            fits file name without the full path) to a specific frame
            type (e.g., arc, bias, etc.).  If None, this is determined
            by the :func:`get_frame_types` method.
        usrdata (:obj:`astropy.table.Table`, optional):
            A user provided set of data used to supplement or overwrite
            metadata read from the file headers.  The table must have a
            `filename` column that is used to match to the metadata
            table generated within PyPhot.
        setups (:obj:`list`, optional):
            A list of setups that each file can be associated with.  If
            None, all files are expected to be for a single setup.
        cfg_lines (:obj:`list`, optional):
            A list of strings that provide a set of user-defined
            parameters for executing pyphot.  These are the lines of a
            configuration file.  See the documentation for the
            `configobj`_ package.  One of the user-level inputs should
            be the camera that provided the data to be reduced.
            One can get the list of cameras currently served by
            running::
                
                from pyphot.cameras.util import valid_cameras
                print(valid_cameras())

            To use all the default parameters when reducing data from a
            given camera, you can set `cfg_lines = None`, but you
            then *must* provide the `camera_name`.
        camera_name (:obj:`str`, optional):
            If not providing a list of configuration lines
            (`cfg_lines`), this sets the camera.  The camera
            defined in `cfg_lines` takes precedent over anything
            provided by this argument.
        pyphot_file (:obj:`str`, optional):
            The name of the pyphot file used to instantiate the
            reduction. This can be None, and will lead to default
            names for output files (TODO: Give list). Setting
            :ref:`pyphot_file` here *only sets the name of the file*.
            To instantiate a :class:`~pyphot.pyphotsetup.PyPhotSetup`
            object directly from a pyphot file (i.e. by reading the
            file), use the :func:`from_pyphot_file` method; i.e.::
                
                setup = PyPhotSetup.from_pyphot_file('myfile.pyphot')

    Attributes:
        file_list (list):
            See description of class argument.
        frametype (dict):
            See description of class argument.
        setups (list):
            See description of class argument.
        pyphot_file (str):
            See description of class argument.
        camera (:class:`pyphot.cameras.camera.Spectrograph`):
            An instance of the `Spectograph` class used throughout the
            reduction procedures.
        par (:class:`pyphot.par.pyphotpar.PypitPar`):
            An instance of the `PypitPar` class that provides the
            parameters to all the algorthms that pyphot uses to reduce
            the data.
        fitstbl (:class:`pyphot.metadata.PyPhotMetaData`):
            A `Table` that provides the salient metadata for the fits
            files to be reduced.
        setup_dict (dict):
            The dictionary with the list of instrument setups.
        steps (list):
            The steps run to provide the pyphot setup.
    """

    def __init__(self, file_list, path=None, frametype=None, usrdata=None, setups=None,
                 cfg_lines=None, camera_name=None, pyphot_file=None):

        # The provided list of files cannot be None
        if file_list is None or len(file_list) == 0:
            msgs.error('Must provide a list of files to be reduced!')

        # Save input
        self.file_list = file_list
        self.path = os.getcwd() if path is None else path
        self.frametype = frametype
        self.usrdata = usrdata
        self.setups = setups
        self.pyphot_file = pyphot_file
        self.user_cfg = cfg_lines

        # Determine the camera name
        _camera_name = camera_name if cfg_lines is None \
                    else PyPhotPar.from_cfg_lines(merge_with=cfg_lines)['rdx']['camera']

        # Cannot proceed without camera name
        if _camera_name is None:
            msgs.error('Must provide camera name directly or using configuration lines.')
       
        # Instantiate the camera
        self.camera = load_camera(_camera_name)

        # Get the camera specific configuration to be merged with
        # the user modifications.
        camera_cfg_lines = self.camera.default_pyphot_par().to_config()

        # Instantiate the pyphot parameters.  The user input
        # configuration (cfg_lines) can be None.
        self.par = PyPhotPar.from_cfg_lines(cfg_lines=camera_cfg_lines, merge_with=cfg_lines)

        # Prepare internals for execution
        self.fitstbl = None
        self.setup_dict = None
        self.steps = []



    @classmethod
    def from_pyphot_file(cls, filename):
        """
        Instantiate the :class:`PypeitSetup` object using a pyphot file.

        Args:
            filename (str):
                Name of the pyphot file to read.  Pypit files have a
                specific set of valid formats. A description can be
                found :ref:`pyphot_file`.
        
        Returns:
            :class:`PypeitSetup`: The instance of the class.
        """
        cfg_lines, data_files, frametype, usrdata, setups = parse_pyphot_file(filename)
        return cls(data_files, frametype=frametype, usrdata=usrdata, setups=setups,
                   cfg_lines=cfg_lines, pyphot_file=filename)

    @classmethod
    def from_file_root(cls, root, camera, extension='.fits', output_path=None):
        """
        Instantiate the :class:`PyPhotSetup` object by providing a file
        root.
        
        This is based on first writing a vanilla PyPhot file for the
        provided camera and extension to a file in the provided
        path.

        Args:
            root (:obj:`str`):
                The root path to all the files for PyPhot to reduce.
                This should be everything up to the wild-card before the
                file extension to use to find the relevant files.  The
                root itself can have wild cards to read through multiple
                directories.
            camera (:obj:`str`):
                The PyPhot name of the camera used to take the
                observations.  This should be one of the available
                options in
                :func:`pyphot.cameras.valid_cameras`.
            extension (:obj:`str`, optional):
                The extension common to all the fits files to reduce.
                Default is '.fits', meaning anything with `root*.fits*`
                will be be included.
            output_path (:obj:`str`, optional):
                Path to use for the output.  If None, the default is
                './setup_files'.  If the path doesn't yet exist, it is
                created.
        
        Returns:
            :class:`PypitSetup`: The instance of the class.
        """
        # Set the output directory
        outdir = os.path.join(os.getcwd(), 'setup_files') if output_path is None else output_path
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # Set the output file name
        date = str(datetime.date.today().strftime('%Y-%b-%d'))
        pyphot_file = os.path.join(outdir, '{0}_{1}.pyphot'.format(camera, date))
        msgs.info('A vanilla pyphot file will be written to: {0}'.format(pyphot_file))
        
        # Generate the pyphot file
        cls.vanilla_pyphot_file(pyphot_file, root, camera, extension=extension)

        # Now setup PyPhot using that file
        return cls.from_pyphot_file(pyphot_file)

    @staticmethod
    def vanilla_pyphot_file(pyphot_file, root, camera, extension='.fits'):
        """
        Write a vanilla PyPhot file.

        Args:
            pyphot_file (str):
              Name of PyPhot file to be generated
            root (str):
            camera (str):
              Name of camera
            extension (str, optional):
              File extension

        Returns:

        """
        # Generate
        dfname = os.path.join(root, '*{0}*'.format(extension)) \
                    if os.path.isdir(root) else '{0}*{1}*'.format(root, extension)
        # configuration lines
        cfg_lines = ['[rdx]']
        cfg_lines += ['    camera          = {0}'.format(camera)]
        cfg_lines += ['    skip_step_one   = False']
        cfg_lines += ['    skip_master     = False']
        cfg_lines += ['    skip_detproc    = False']
        cfg_lines += ['    skip_sciproc    = False']
        cfg_lines += ['    skip_step_two   = False']
        cfg_lines += ['    skip_astrometry = False']
        cfg_lines += ['    skip_chipcal    = False']
        cfg_lines += ['    skip_img_qa     = False']
        cfg_lines += ['    skip_coadd      = False']
        cfg_lines += ['    skip_zptcal     = False']
        cfg_lines += ['    skip_detection  = False']
        cfg_lines += ['    n_process       = 8']
#        cfg_lines += ['    sortroot = {0}'.format(root)]
        make_pyphot_file(pyphot_file, camera, [dfname], cfg_lines=cfg_lines, setup_mode=True)

    @property
    def nfiles(self):
        """The number of files to reduce."""
        if self.fitstbl is None:
            msgs.warn('No fits files have been read!')
        return 0 if self.fitstbl is None else len(self.fitstbl)

    def __repr__(self):
        return '<{:s}: nfiles={:d}>'.format(self.__class__.__name__, self.nfiles)

    def build_fitstbl(self, strict=True):
        """
        Construct the table with metadata for the frames to reduce.

        Largely a wrapper for :func:`pyphot.core.load.create_fitstbl`.

        Args:
            strict (:obj:`bool`, optional):
                Function will fault if :func:`fits.getheader` fails to
                read the headers of any of the files in
                :attr:`file_list`.  Set to False to only report a
                warning and continue.

        Returns:
            :obj:`astropy.table.Table`: Table with the metadata for each
            fits file to reduce.  Note this is different from
            :attr:`fitstbl` which is a :obj:`PyPhotMetaData` object
        """
        # Build and sort the table
        self.fitstbl = PyPhotMetaData(self.camera, par=self.par, files=self.file_list,
                                      usrdata=self.usrdata, strict=strict)
        # Sort by the time
        if 'time' in self.fitstbl.keys():
            self.fitstbl.sort('time')

        # Add this to the completed steps
        self.steps.append(inspect.stack()[0][3])

        # Return the table
        return self.fitstbl.table

    def get_frame_types(self, flag_unknown=False, use_header_id=False):
        """
        Include the frame types in the metadata table.

        This is mainly a wrapper for
        :func:`PyPhotMetaData.get_frame_types`.

        .. warning::

            Because this merges the frame types with the existing
            :attr:`fitstbl` this should only be run once.

        Args:
            flag_unknown (:obj:`bool`, optional):
                Allow for frames to have unknown types instead of
                crashing.  This should be True for initial setup and
                False otherwise.  Passed to get_frame_types()
            use_header_id (bool, optional):
                Passed to get_frame_types()

        """
        # Use PyPhotMetaData methods to get the frame types
        _ = self.fitstbl.get_frame_types(flag_unknown=flag_unknown, user=self.frametype,
                                         useIDname=use_header_id)
        # Include finished processing step
        self.steps.append(inspect.stack()[0][3])

    def load_metadata(self, fits_file):
        """
        Load the fitstbl from disk (a binary FITS table)

        Args:
            fits_file (str):  Name of PyPhotMetaData file

        Returns:
            obj:`PyPhotMetaData`: The so-called fitstbl

        """
        self.fitstbl = PyPhotMetaData(self.camera, self.par, data=Table.read(fits_file))
        msgs.info("Loaded fitstbl from {:s}".format(fits_file))
        return self.fitstbl.table

    def write_metadata(self, sort_dir=None, ofile=None):
        """
        Write the `astropy.table.Table`_ object in :attr:`fitstbl`
        to a file.

        If an output file is provided, the file is used.  If that file
        name inclues '.fits', the output will be a fits file; otherwise
        the format is ascii.fixed_width.

        If no output file, the default is an ascii table with an '.lst'
        extension.  The root name of the file is either the camera
        name or the root name of the pyphot file, if the latter exists.

        If a `sort_dir` is provided, the directory of the nominal output
        file is changed to this directory.

        Args:
            sort_dir (:obj:`str`, optional):
                The full root of the name for the metadata table
                ('.lst') file.
            ofile (:obj:`str`, optional):
                The name of the file to write.  See description above.
        """
        if ofile is None:
            ofile = self.camera.name + '.lst' if self.pyphot_file is None \
                        else self.pyphot_file.replace('.pyphot', '.lst')
            if sort_dir is not None:
                ofile = os.path.join(sort_dir, os.path.split(ofile)[1])

        format = None if '.fits' in ofile else 'ascii.fixed_width'
        self.fitstbl.write(ofile,
                           #columns=None if format is None else self.camera.pyphot_file_keys(),
                           format=format, overwrite=True)

    # TODO: Check if user_header_id is ever actually used.
    def run(self, setup_only=False, use_header_id=False, sort_dir=None,
            clean_config=True):
        """
        Once instantiated, this is the main method used to construct the
        object.
        
        The code flow is as follows::
            - Build the fitstbl from an input file_list (optional)
            - Type the files (bias, arc, etc.)
            - Match calibration files to the science files
            - Generate the setup_dict
                - Write group info to disk
                - Write calib info to disk (if main run)

        It is expected that a user will run this three times if they're
        being careful.  Once with `setup_only=True` to confirm the
        images are properly typed and grouped together for calibration.
        A second time with `calibration_check=True` to confirm the
        appropriate calibrations frames are available.  And a third time
        to do the actual setup before proceeding with the reductions.

        Args:
            setup_only (:obj:`bool`, optional):
                Only this setup will be performed. ``PyPhot`` is
                expected to execute in a way that ends after this
                class is fully instantiated such that the user can
                inspect the results before proceeding. This has the
                effect of providing more output describing the
                success of the setup and how to proceed, and provides
                warnings (instead of errors) for issues that may
                cause the reduction itself to fail.
            use_header_id (:obj:`bool`, optional):
                Allow setup to use the frame types drawn from single,
                instrument-specific header keywords set to ``idname``
                in the metadata table (:attr:`fitstbl`).
            sort_dir (:obj:`str`, optional):
                The directory to put the '.sorted' file.
            clean_config (:obj:`bool`, optional):
                Remove files with metadata that indicate an
                instrument configuration that ``PyPhot`` cannot
                reduce. See
                :func:`~pyphot.cameras.camera.Spectrograph.valid_configuration_values`.

        Returns:
            :obj:`tuple`: Returns, respectively, the
            :class:`~pyphot.par.pyphotpar.PyPhotPar` object with the
            reduction parameters, the
            :class:`~pyphot.cameras.camera.Spectrograph`
            object with the camera instance, and an
            `astropy.table.Table`_ with the frame metadata. If
            ``setup_only`` is True, these are all returned as None
            values.
        """
        # Kludge
        pyphot_file = '' if self.pyphot_file is None else self.pyphot_file

        # Build fitstbl
        if self.fitstbl is None:
            self.build_fitstbl(strict=not setup_only)#, bkg_pairs=bkg_pairs)

        # Remove frames that have invalid values for
        # configuration-defining metadata
        if clean_config:
            self.fitstbl.clean_configurations()

        # File typing
        self.get_frame_types(flag_unknown=setup_only,
                             use_header_id=use_header_id)

        # Determine the configurations and assign each frame to the
        # specified configuration
        cfgs = self.fitstbl.unique_configurations()
        self.fitstbl.set_configurations(cfgs)

        # Assign frames to calibration groups
        self.fitstbl.set_calibration_groups() #global_frames=['bias', 'dark'])

        # Set default coadd_id
        self.fitstbl.set_combination_groups()

        # TODO: Are we planning to do this?
        # Assign science IDs based on the calibrations groups (to be
        # deprecated)
        self.fitstbl['failures'] = False                    # TODO: placeholder

        if setup_only:
            # Collate all matching files and write .sorted Table (on pyphot_setup only)
            sorted_file = self.camera.name + '.sorted' \
                                if pyphot_file is None or len(pyphot_file) == 0 \
                                else pyphot_file.replace('.pyphot', '.sorted')
            if sort_dir is not None:
                sorted_file = os.path.join(sort_dir, os.path.split(sorted_file)[1])
            self.fitstbl.write_sorted(sorted_file)
            msgs.info("Wrote sorted file data to {:s}".format(sorted_file))

        else:
            # Write the calib file
            # This is currently needed for QA
            calib_file = self.camera.name + '.calib' \
                                if pyphot_file is None or len(pyphot_file) == 0 \
                                else pyphot_file.replace('.pyphot', '.calib')
            if sort_dir is not None:
                calib_file = os.path.join(sort_dir, os.path.split(calib_file)[1])
            self.fitstbl.write_calib(calib_file)

        # Finish
        if setup_only:
            for idx in np.where(self.fitstbl['failures'])[0]:
                msgs.warn("No Arc found: Skipping object {:s} with file {:s}".format(
                            self.fitstbl['target'][idx],self.fitstbl['filename'][idx]))
            msgs.info("Setup is complete.")
            msgs.info("Inspect the .sorted file")
            return None, None, None

        return self.par, self.camera, self.fitstbl


