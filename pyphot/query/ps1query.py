
import sys, os
import json
import requests

from astropy.io import ascii
from astropy.table import Table


from urllib.parse import quote as urlencode
import http.client as httplib

from pyphot import msgs

def conesearch(ra, dec, radius, table="mean", release="dr2", format="csv", columns=None,
              savetable=None, verbose=False, **kw):
    """Do a cone search of the PS1 catalog

    Parameters
    ----------
    ra (float): (degrees) J2000 Right Ascension
    dec (float): (degrees) J2000 Declination
    radius (float): (degrees) Search radius (<= 0.5 degrees)
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2)

    Example:
      results = conesearch(106.60991276,29.35151732, 1.0/60., table="mean", release="dr1", format="csv")

    """

    data = kw.copy()
    data['ra'] = ra
    data['dec'] = dec
    data['radius'] = radius
    if format in ['votable', 'json']:
        results = query(table=table, release=release, format=format, columns=columns,
                        verbose=verbose, **data)
    elif format in ['csv', 'fits']:
        results = query(table=table, release=release, format='csv', columns=columns,
                        verbose=verbose, **data)
    else:
        msgs.error("Format {:} does not support! Please choose the following format: 'fits', 'csv', 'votable', 'json'")

    if savetable is not None:
        if format in ['csv', 'fits']:
            if len(results) == 0:
                msgs.warn("Nothing found for {:} {:}".format(ra, dec))
            else:
                results.write('{:}.cat.fits'.format(savetable), overwrite=True, format=format)
        else:
            msgs.error("Please use format csv or fits in order to save the query results.")

    return results

def query(table="mean", release="dr2", format="csv", columns=None, verbose=False, **kw):
    """Do a general search of the PS1 catalog (possibly without ra/dec/radius)

    Parameters
    ----------
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2).  Note this is required!
    """

    data = kw.copy()
    if not data:
        msgs.error("You must specify some parameters for search")
    checklegal(table, release)
    if format not in ("csv", "votable", "json"):
        msgs.error("Bad value for format")
    if columns:
        # check that column values are legal
        # create a dictionary to speed this up
        dcols = {}
        for col in metadata(table, release)['name']:
            dcols[col.lower()] = 1
        badcols = []
        for col in columns:
            if col.lower().strip() not in dcols:
                badcols.append(col)
        if badcols:
            msgs.error('Some columns not found in table: {}'.format(', '.join(badcols)))
        # two different ways to specify a list of column values in the API
        # data['columns'] = columns
        data['columns'] = '[{}]'.format(','.join(columns))

    # submit the request
    try:
        baseurl = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"
        url = "{baseurl}/{release}/{table}.{format}".format(**locals())
        r = requests.get(url, params=data)
        r.raise_for_status()
    except:
        if 'radius' in data.keys():
            data['radius'] *= 60.0 # old API using arcmin as default
        msgs.warn('NEW API does not respond, using the old one')
        url = "https://archive.stsci.edu/panstarrs/search.php?action=Search&outputformat={:}&coordformat=dec".format(format.upper())
        r = requests.get(url, params=data)
        r.raise_for_status()

    if verbose:
        print(r.url)
    if format == "json":
        return r.json()
    else:
        try:
            tab = ascii.read(r.text)
            if "Ang Sep (')" in tab.keys():
                # OLD data need some work
                tab['distance'] = tab["Ang Sep (')"]
                tab.remove_column("Ang Sep (')")
                tab.remove_row(0)
            tab.write('tmp.csv')
            tab = Table.read('tmp.csv')
            os.system('rm tmp.csv')
        except:
            msgs.warn('Nothing found!')
            tab = None
        return tab


def checklegal(table, release):
    """Checks if this combination of table and release is acceptable

    Raises a VelueError exception if there is problem
    """

    releaselist = ("dr1", "dr2")
    if release not in ("dr1", "dr2"):
        msgs.error("Bad value for release (must be one of {})".format(', '.join(releaselist)))
    if release == "dr1":
        tablelist = ("mean", "stack")
    else:
        tablelist = ("mean", "stack", "detection")
    if table not in tablelist:
        msgs.error("Bad value for table (for {} must be one of {})".format(release, ", ".join(tablelist)))


def metadata(table="mean", release="dr1"):
    """Return metadata for the specified catalog and table

    Parameters
    ----------
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    Returns an astropy table with columns name, type, description
    """

    # baseurl for PS1 API
    baseurl = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"

    checklegal(table, release)
    url = "{baseurl}/{release}/{table}/metadata".format(**locals())
    r = requests.get(url)
    r.raise_for_status()
    v = r.json()
    # convert to astropy table
    tab = Table(rows=[(x['name'], x['type'], x['description']) for x in v],
                names=('name', 'type', 'description'))
    return tab


def mastQuery(request):
    """Perform a MAST query.

    Parameters
    ----------
    request (dictionary): The MAST request json object

    Returns head,content where head is the response HTTP headers, and content is the returned data
    """

    server = 'mast.stsci.edu'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent": "python-requests/" + version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request=" + requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head, content
