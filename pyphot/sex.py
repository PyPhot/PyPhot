import os,re,subprocess
import numpy as np
from pyphot import msgs
from pkg_resources import resource_filename
config_dir = resource_filename('pyphot', '/config/')

defaultparams = ['NUMBER','X_IMAGE', 'Y_IMAGE','XWIN_IMAGE','YWIN_IMAGE','ERRAWIN_IMAGE','ERRBWIN_IMAGE',
                 'ERRTHETAWIN_IMAGE','ALPHA_J2000', 'DELTA_J2000','ISOAREAF_IMAGE','ISOAREA_IMAGE','ELLIPTICITY','ELONGATION',
                 'KRON_RADIUS','FWHM_IMAGE','CLASS_STAR','FLAGS',
                 'BACKGROUND','FLUX_MAX','MAG_ISO','MAGERR_ISO',
                 'FLUX_ISO','FLUXERR_ISO','MAG_AUTO','MAGERR_AUTO','FLUX_AUTO','FLUXERR_AUTO','MAG_BEST',
                 'MAGERR_BEST','FLUX_BEST','FLUXERR_BEST','FLUX_RADIUS','MAG_APER(5)','MAGERR_APER(5)',
                 'FLUX_APER(5)','FLUXERR_APER(5)']

def get_version(task='sex'):
    """
    To find the SExtractor version
    returns: a string (e.g. '2.4.4')
    """
    v = subprocess.Popen(task, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = v.communicate()
    version_match = re.search("[Vv]ersion ([0-9\.])+", err.decode("utf-8"))
    version = str(version_match.group()[8:])
    return version


def get_default_config(task='sex', defaultconfig='pyphot', workdir='./', verbose=True):
    """
    To get the default SExtractor configuration file
    """

    if defaultconfig == "sex":
        p = subprocess.Popen([task, "-d"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        f = open(workdir + "config.sex", "w")
        f.write(out)
        f.close()
        if verbose:
            msgs.info("config.sex generated from SExtractor default configuration")

    elif defaultconfig == "pyphot":
        os.system("cp " + os.path.join(config_dir,"sex.config") + ' ' + os.path.join(workdir,"config.sex"))
        if verbose:
            msgs.info("config.sex generated from PyPhot default configuration")
    else:
        os.system("cp " + defaultconfig + ' ' + os.path.join(workdir,"config.sex"))
        if verbose:
            msgs.info("Using user provided configuration for SExtractor")

    comd = ["-c", os.path.join(workdir,"config.sex")]
    return comd


def get_config(config=None, workdir="./", dual=False):
    """
    append some configure parameters to the default configure file
    e.g. config = {"DETECT_MINAREA":10, "PHOT_APERTURES":"5, 10, 20, 30","CATALOG_TYPE":"FITS_LDAC",
                    "MAG_ZEROPOINT":24.5}
    """

    if "WEIGHT_IMAGE" in config:
        if dual:
            weights = config["WEIGHT_IMAGE"].split(',')
            weight1 = os.path.join(workdir, weights[0])
            weight2 = os.path.join(workdir, weights[1])
            config["WEIGHT_IMAGE"] = weight1 + "," + weight2
        else:
            config["WEIGHT_IMAGE"] = os.path.join(workdir, config["WEIGHT_IMAGE"])
    if "FLAG_IMAGE" in config:
        if dual:
            weights = config["FLAG_IMAGE"].split(',')
            weight1 = os.path.join(workdir, weights[0])
            weight2 = os.path.join(workdir, weights[1])
            config["FLAG_IMAGE"] = weight1 + "," + weight2
        else:
            config["FLAG_IMAGE"] = os.path.join(workdir, config["FLAG_IMAGE"])
    if "CATALOG_NAME" in config:
        config["CATALOG_NAME"] = os.path.join(workdir, config["CATALOG_NAME"])
    if "CHECKIMAGE_NAME" in config:
        config["CHECKIMAGE_NAME"] = os.path.join(workdir, config["CHECKIMAGE_NAME"])
    if "XML_NAME" in config:
        config["XML_NAME"] = os.path.join(workdir, config["XML_NAME"])

    if config == None:
        configapp = []
    else:
        configapp = []
        for (key, value) in config.items():
            configapp.append("-" + str(key))
            if (key == 'PHOT_APERTURES') and (np.size(config['PHOT_APERTURES'])>1):
                separator = ','
                configapp.append(separator.join(config['PHOT_APERTURES'].astype('str')))
            else:
                configapp.append(str(value).replace(' ', ''))

    return configapp


def get_nnw(nnw=None, workdir='./', verbose=True):
    """
    To get the default SExtractor configuration file
    """
    if (nnw == None) or (nnw=='sex'):
        os.system("cp " + os.path.join(config_dir,"sex.nnw") + ' ' + os.path.join(workdir,"nnw.sex"))
        if verbose:
            msgs.info("nnw.sex generated from PyPhot default NNW")
    else:
        os.system("cp " + nnw + ' ' + os.path.join(workdir,"nnw.sex"))
        if verbose:
            msgs.info("Using user provided NNW")

    comd = ["-STARNNW_NAME", os.path.join(workdir,"nnw.sex")]
    return comd


def get_conv(conv=None, workdir='./', verbose=True):
    """
    Get the default convolution matrix, if needed.
    """
    if (conv == None) or (conv == "sex"):
        os.system("cp " + os.path.join(config_dir,"sex.conv") + ' ' + os.path.join(workdir,"conv.sex"))
        if verbose:
            msgs.info("conv.sex using 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels")
    elif conv == "sex995":
        os.system("cp " + os.path.join(config_dir, "sex995.conv") + ' ' + os.path.join(workdir, "conv.sex"))
        if verbose:
            msgs.info("conv.sex using 9x9 convolution mask of a gaussian PSF with FWHM = 5.0 pixels")
    else:
        os.system("cp " + conv + ' ' + os.path.join(workdir,"conv.sex"))
        if verbose:
            msgs.info("Using user provided conv")

    comd = ["-FILTER_NAME", os.path.join(workdir,"conv.sex")]
    return comd

def get_params(params=None, workdir='./'):
    """
    To write the SExtractor paramaters into a file
    """

    if params == None:
        f = open(os.path.join(workdir,"params.sex"),"w")
        f.write("\n".join(defaultparams))
        f.write("\n")
        f.close()
    else:
        try:
            os.system("cp " + params +' ' + os.path.join(workdir,"params.sex"))
            msgs.info("Using user provided params file")
        except:
            f = open(os.path.join(workdir,"params.sex"),"w")
            f.write("\n".join(params))
            f.write("\n")
            f.close()

    comd = ["-PARAMETERS_NAME", os.path.join(workdir,"params.sex")]
    return comd


def sexone(imgname, task='sex', config=None, workdir='./', params=None, defaultconfig='pyphot',
           conv=None, nnw=None, dual=False, flag_image=None, weight_image=None,
           delete=True, log=False, verbose=True):

    if verbose:
        ## Get the version of your SExtractor
        sexversion = get_version(task=task)
        msgs.info("SExtractor version is {:}".format(sexversion))

    ## Generate the configuration file
    configcomd = get_default_config(defaultconfig=defaultconfig, workdir=workdir, verbose=verbose)

    ## Generate the convolution matrix
    convcomd = get_conv(conv=conv, workdir=workdir, verbose=verbose)

    ## Generate the NNW file
    nnwcomd = get_nnw(nnw=nnw, workdir=workdir, verbose=verbose)

    ## Generate the parameters file
    paramscomd = get_params(params=params, workdir=workdir)

    ## append your configuration
    if config is None:
        config = {"CHECKIMAGE_TYPE": "NONE", "WEIGHT_TYPE": "NONE", "CATALOG_NAME": "dummy.cat",
                  "CATALOG_TYPE": "FITS_LDAC", "BACK_TYPE ": "MANUAL", "BACK_VALUE": 0.0}
    config['CATALOG_NAME'] = imgname.replace('.fits','_cat.fits')
    if flag_image is not None:
        config['FLAG_IMAGE'] = flag_image
    if weight_image is not None:
        config['WEIGHT_IMAGE'] = weight_image

    configapp = get_config(config=config, workdir=workdir, dual=dual)

    if dual:
        comd = [task] + [os.path.join(workdir,imgname[0]) + "," + os.path.join(workdir,imgname[1])] +\
               configcomd + convcomd + nnwcomd + paramscomd + configapp
    else:
        comd = [task] + [os.path.join(workdir,imgname)] + configcomd + convcomd + nnwcomd + paramscomd + configapp
    p = subprocess.Popen(comd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if verbose:
        msgs.info("Catalog generated: " + config['CATALOG_NAME'])

    if log:
        if dual:
            logfile = open(os.path.join(workdir, imgname[1][:-5]+".sex.log"), "w")
        else:
            logfile = open(os.path.join(workdir, imgname[:-5]+".sex.log"), "w")
        logfile.write("SExtractor was called with :\n")
        logfile.write(" ".join(comd))
        logfile.write("\n\n####### stdout #######\n")
        logfile.write(out.decode("utf-8"))
        logfile.write("\n####### stderr #######\n")
        logfile.write(err.decode("utf-8"))
        logfile.write("\n")
        logfile.close()
        if verbose:
            msgs.info("Processing log generated: " + os.path.join(workdir, imgname[:-5] + ".sex.log"))
    if delete:
        os.system("rm {:}".format(os.path.join(workdir,"*.sex")))

def sexall(imglist, task='sex', config=None, workdir='./', params=None, defaultconfig='pyphot',
           conv=None, nnw=None, dual=False, flag_image_list=None, weight_image_list=None,
           delete=True, log=False):

    if flag_image_list is not None:
        assert len(imglist) == len(flag_image_list), "flag_image_list should have the same length with imglist"
    if weight_image_list is not None:
        assert len(imglist) == len(weight_image_list), "weight_image_list should have the same length with imglist"

    for ii, imgname in enumerate(imglist):
        msgs.info('Extracting photometric catalog with SExtractor {:} for {:}'.format(get_version(),os.path.basename(imgname)))
        if flag_image_list is not None:
            flag_image = flag_image_list[ii]
        else:
            flag_image = None
        if weight_image_list is not None:
            weight_image = weight_image_list[ii]
        else:
            weight_image = None
        if config is not None:
            this_config = config.copy()# need to copy this since the config would be possibly changed in sexone!
        else:
            this_config = None
        sexone(imgname, task=task, config=this_config, workdir=workdir, params=params, defaultconfig=defaultconfig, conv=conv,
               nnw=nnw, dual=dual, flag_image=flag_image, weight_image=weight_image, delete=delete, log=log, verbose=False)

