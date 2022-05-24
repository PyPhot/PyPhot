0.5.0dev (2022-05-24)
----------------
- Add variance map which includes photon noise
- Change weight map from 1/BKG to 1/(FLAG*SKYCOUNTS)
- Change ccdproc to detproc and add new ImageProc class
- Add Astrometry class
- Support external reference catalog for scamp (with datalab requirement if use)
- Simplify pyphot.py

0.4.0 (2022-02-21)
----------------
- Major development for Keck LRIS
- Support detectors with multiple amplifiers
- Improve the overall flowcharting of pyphot.py
- More step-control parameters: skip_master, skip_ccdproc, skip_sciproc, skip_astrometry
- Switch the astrometry mode from calibrating chips individually to calibrating everything in a group
- Minor improvement on master flatframe building
- MaterFrame is a class and supports doing calibrations without science frames

0.3.0 (2022-02-21)
----------------
- Modify README.md on software installation
- Add this file and tag the previous versions
- The final version of doing astrometric calibrations chip-by-chip
- Support multiprocessing
- Some tweaks on Magellan IMACS
- Reduced IMACS data for J0525 field

0.2.0 (2021-07-10)
----------------
- First science-ready pipeline for IMACS
- Major steps on Magellan IMACS data reduction pipeline
- Reduced IMACS data for J1526 field

0.1.0 (2021-06-14)
----------------
- First release of PyPhot

0.0.1 (2021-06-02)
----------------
- First science-ready pipeline for LBC and WIRCam
- Major steps on LBT LBC data reduction pipeline
- Major steps on CFHT WIRCam data reduction pipeline
- Reduced both WIRCam and LBC data for J0100 quasar field
- Photometrically calibrating individual chips/exposures

0.0.0 (2020-12-14)
----------------

Start development of PyPhot