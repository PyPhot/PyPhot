"""
ToDo: Improve it and change to a class
"""

import os, re, subprocess

import multiprocessing
from multiprocessing import Process, Queue

from pyphot import msgs
from pkg_resources import resource_filename
config_dir = resource_filename('pyphot', '/config/')


def get_version():
    """
    To find the SWARP version
    returns: a string (e.g. '2.4.4')
    """
    v = subprocess.Popen("swarp", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = v.communicate()
    version_match = re.search("[Vv]ersion ([0-9\.])+", err.decode("utf-8"))
    version = str(version_match.group()[8:])
    return version


def get_default_config(defaultconfig='pyphot', workdir='./', outroot='pyphot', verbose=True):
    """
    To get the default SWARP configuration file
    """

    if defaultconfig == "swarp":
        p = subprocess.Popen(["swarp", "-d"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        f = open(os.path.join(workdir, outroot+"_config.swarp"), "w")
        f.write(out)
        f.close()
        if verbose:
            msgs.info("config.swarp generated from SWarp default configuration")
    elif defaultconfig == "pyphot":
        os.system("cp " + os.path.join(config_dir,"swarp.config") + ' ' + os.path.join(workdir,outroot+"_config.swarp"))
        if verbose:
            msgs.info("config.swarp generated from PyPhot default configuration")
    else:
        os.system("cp " + defaultconfig + ' ' + os.path.join(workdir,outroot+"_config.swarp"))
        if verbose:
            msgs.info("Using user provided configuration for SWarp")

    comd = ["-c", os.path.join(workdir, outroot+"_config.swarp")]
    return comd


def get_config(config=None):
    """
    append some configure parameters to the default configure file
    e.g. config = {"WEIGHT_TYPE":"MAP_WEIGHT", "COMBINE":"Y", "COMBINE_TYPE":"WEIGHTED"}
    """

    if config == None:
        configapp = []
    else:
        configapp = []
        for (key, value) in config.items():
            configapp.append("-" + str(key))
            configapp.append(str(value).replace(' ', ''))

    return configapp


def swarpone(imgname, config=None, workdir='./', defaultconfig='pyphot', delete=True, log=False, verbose=True):

    if verbose:
        ## Get the version of your swarp
        swarpversion = get_version()
        msgs.info("SWarp version is {:}".format(swarpversion))

    imgroot = imgname.replace('.fits','')

    ## Generate the configuration file
    configcomd = get_default_config(defaultconfig=defaultconfig, workdir=workdir, outroot=imgroot, verbose=verbose)

    ## append your configuration
    ## ToDO: Not sure why set the weightout_name does not work
    #config['WEIGHTOUT_NAME'] = imgname.replace('.fits','.wht.fits')
    configapp = get_config(config=config)

    comd = ["swarp"] + [os.path.join(workdir, imgname)] + configcomd + configapp + \
           ["-COMBINE"] + ["N"] + ["-RESAMPLE_DIR"] + [workdir] + \
           ["-XML_NAME"] + [os.path.join(workdir, imgname.replace('.fits','.swarp.xml'))]

    p = subprocess.Popen(comd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    if log:
        logfile = open(os.path.join(workdir, imgname[:-5]+".swarp.log"), "w")
        logfile.write("swarp was called with :\n")
        logfile.write(" ".join(comd))
        logfile.write("\n\n####### stdout #######\n")
        logfile.write(out.decode("utf-8"))
        logfile.write("\n####### stderr #######\n")
        logfile.write(err.decode("utf-8"))
        logfile.write("\n")
        logfile.close()
        if verbose:
            msgs.info("Processing log generated: " + os.path.join(workdir, imgname[:-5]+".swarp.log"))
    if delete:
        os.system("rm " + os.path.join(workdir, imgroot+"_*.swarp"))
        if os.path.exists(os.path.join(workdir, imgname.replace('.fits','.swarp.xml'))):
            os.system("rm {:}".format(os.path.join(workdir, imgname.replace('.fits','.swarp.xml'))))

def _swarpone_worker(work_queue, config=None, workdir='./', defaultconfig='pyphot', delete=True, log=False, verbose=True):

    """Multiprocessing worker for sexone."""

    if config is not None:
        this_config = config.copy()  # need to copy this since the config would be possibly changed in sexone!
    else:
        this_config = None

    while not work_queue.empty():
        imgname = work_queue.get()
        swarpone(imgname, config=config, workdir=workdir, defaultconfig=defaultconfig, delete=delete, log=log, verbose=verbose)

def run_swarp(imglist, config=None, workdir='./', defaultconfig='pyphot', coadddir=None, coaddroot=None,
              n_process=4, delete=False, log=False, verbose=False):

    if coaddroot is not None:
        if coadddir is None:
            coadddir = os.path.join(workdir, "Coadd")
        if not os.path.exists(coadddir):
            os.mkdir(coadddir)

        if verbose:
            msgs.info("Coadded images are stored at {:}".format(coadddir))

        ## Generate a tmp list to store the imagename with path
        tmplist = open(os.path.join(workdir, "tmplist.txt"), "w")
        for img in imglist:
            print(img, file=tmplist)
        tmplist.close()

        ## Get the version of your swarp
        swarpversion = get_version()
        if verbose:
            msgs.info("SWarp version is {:}".format(swarpversion))

        ## Generate the configuration file
        configcomd = get_default_config(defaultconfig=defaultconfig, workdir=coadddir, outroot=coaddroot, verbose=verbose)

        ## append your configuration
        configapp = get_config(config=config)
        comd = ["swarp"] + ["@" + os.path.join(workdir, "tmplist.txt")] + configcomd + configapp + \
               ["-COMBINE"] + ["Y"] + ["-IMAGEOUT_NAME"] + [os.path.join(coadddir, coaddroot + ".fits")] + \
               ["-WEIGHTOUT_NAME"] + [os.path.join(coadddir, coaddroot + ".weight.fits")] + \
               ["-RESAMPLE_DIR"] + [coadddir] + ["-XML_NAME"] + [os.path.join(coadddir, coaddroot + ".swarp.xml")]

        # Set the max number of opened files by your computer
        os.system("ulimit -n 4096")

        p = subprocess.Popen(comd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        os.system("rm {:}".format(os.path.join(workdir, "tmplist.txt")))

        if log:
            logfile = open(os.path.join(coadddir, coaddroot+".swarp.log"), "w")
            logfile.write("swarp was called with :\n")
            logfile.write(" ".join(comd))
            logfile.write("\n\n####### stdout #######\n")
            logfile.write(out.decode("utf-8"))
            logfile.write("\n####### stderr #######\n")
            logfile.write(err.decode("utf-8"))
            logfile.write("\n")
            logfile.close()
            msgs.info("Processing log generated: " + os.path.join(coadddir, coaddroot+".swarp.log"))
        if delete:
            os.system("rm " + os.path.join(coadddir, coaddroot+"_config.swarp"))
            if os.path.exists(os.path.join(coadddir, coaddroot + ".swarp.xml")):
                os.system("rm " + os.path.join(coadddir, coaddroot + ".swarp.xml"))

    else:
        if n_process==1:
            for imgname in imglist:
                msgs.info('Resampling {:} with Swarp {:}'.format(os.path.basename(imgname), get_version()))
                if config is not None:
                    this_config = config.copy()# need to copy this since the config would be possibly changed in swarpone!
                else:
                    this_config = None
                swarpone(imgname, config=this_config, workdir=workdir, defaultconfig=defaultconfig,
                         delete=delete, log=log, verbose=False)
        else:
            msgs.info('Start parallel processing with n_process={:}'.format(n_process))
            work_queue = Queue()
            processes = []

            for ii in range(len(imglist)):
                work_queue.put(imglist[ii])

            # creating processes
            for w in range(n_process):
                p = Process(target=_swarpone_worker, args=(work_queue,), kwargs={
                    'config': config, 'workdir': workdir, 'defaultconfig': defaultconfig,
                    'delete': delete, 'log': log, 'verbose': verbose})
                processes.append(p)
                p.start()

            # completing process
            for p in processes:
                p.join()


