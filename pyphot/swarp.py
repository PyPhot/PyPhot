
import os, re, subprocess

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


def get_default_config(defaultconfig='pyphot', workdir='./'):
    """
    To get the default SWARP configuration file
    """

    if defaultconfig == "swarp":
        p = subprocess.Popen(["swarp", "-d"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        f = open(os.path.join(workdir, "config.swarp"), "w")
        f.write(out)
        f.close()
        msgs.info("config.swarp generated from SWarp default configuration")
    elif defaultconfig == "pyphot":
        os.system("cp " + os.path.join(config_dir,"swarp.config") + ' ' + os.path.join(workdir,"config.swarp"))
        msgs.info("config.swarp generated from PyPhot default configuration")
    else:
        os.system("cp " + defaultconfig + ' ' + os.path.join(workdir,"config.swarp"))
        msgs.info("Using user provided configuration for SWarp")

    comd = ["-c", os.path.join(workdir, "config.swarp")]
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


def swarpone(imgname, config=None, workdir='./', defaultconfig='pyphot', delete=True, log=False):

    ## Get the version of your swarp
    swarpversion = get_version()
    msgs.info("SWarp version is {:}".format(swarpversion))

    ## Generate the configuration file
    configcomd = get_default_config(defaultconfig=defaultconfig, workdir=workdir)

    ## append your configuration
    ## ToDO: Not sure why set the weightout_name does not work
    #config['WEIGHTOUT_NAME'] = imgname.replace('.fits','.wht.fits')
    configapp = get_config(config=config)

    comd = ["swarp"] + [os.path.join(workdir, imgname)] + configcomd + configapp + \
           ["-COMBINE"] + ["N"] + ["-RESAMPLE_DIR"] + [workdir]

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
        msgs.info("Processing log generated: " + os.path.join(workdir, imgname[:-5]+".swarp.log"))
    if delete:
        os.system("rm " + os.path.join(workdir,"*.swarp"))
        if os.path.exists(os.path.join(workdir,"swarp.xml")):
            os.system("rm " + os.path.join(workdir,"swarp.xml"))


def swarpall(imglist, config=None, workdir='./', defaultconfig='pyphot', coadddir=None, coaddroot=None, delete=False, log=False):

    if coaddroot is not None:

        if coadddir is None:
            coadddir = os.path.join(workdir, "Coadd")
        if not os.path.exists(coadddir):
            os.mkdir(coadddir)
        msgs.info("Coadded images are stored at {:}".format(coadddir))

        ## Generate a tmp list to store the imagename with path
        tmplist = open(os.path.join(workdir, "tmplist.txt"), "w")
        for img in imglist:
            print(img, file=tmplist)
        tmplist.close()

        ## Get the version of your swarp
        swarpversion = get_version()
        msgs.info("SWarp version is {:}".format(swarpversion))

        ## Generate the configuration file
        configcomd = get_default_config(defaultconfig=defaultconfig, workdir=workdir)

        ## append your configuration
        configapp = get_config(config=config)
        comd = ["swarp"] + ["@" + os.path.join(workdir, "tmplist.txt")] + configcomd + configapp + \
               ["-COMBINE"] + ["Y"] + ["-IMAGEOUT_NAME"] + [os.path.join(coadddir, coaddroot + ".fits")] + \
               ["-WEIGHTOUT_NAME"] + [os.path.join(coadddir, coaddroot + ".weight.fits")] + \
               ["-RESAMPLE_DIR"] + [coadddir] + ["-XML_NAME"] + [os.path.join(coadddir, coaddroot + ".swarp.xml")]

        # Set the max number of opened files by your computer
        os.system("ulimit -n 2048")

        p = subprocess.Popen(comd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        os.system("rm {:}".format(os.path.join(workdir, "tmplist.txt")))

        if log:
            logfile = open(os.path.join(coadddir, coaddroot+".scamp.log"), "w")
            logfile.write("swarp was called with :\n")
            logfile.write(" ".join(comd))
            logfile.write("\n\n####### stdout #######\n")
            logfile.write(out.decode("utf-8"))
            logfile.write("\n####### stderr #######\n")
            logfile.write(err.decode("utf-8"))
            logfile.write("\n")
            logfile.close()
            msgs.info("Processing log generated: " + os.path.join(coadddir, coaddroot+".scamp.log"))
        if delete:
            os.system("rm " + os.path.join(workdir, "*.swarp"))
            if os.path.exists(os.path.join(workdir, "swarp.xml")):
                os.system("rm " + os.path.join(workdir, "swarp.xml"))

    else:
        for imgname in imglist:
            swarpone(imgname,config=config, workdir=workdir, defaultconfig=defaultconfig, delete=delete, log=log)
