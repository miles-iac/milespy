import os
import sys
import glob
import h5py
# import scipy.interpolate
import scipy.ndimage
import numpy                 as np
import matplotlib.pyplot     as plt
from   astropy.io            import fits, ascii
from   scipy.interpolate   import interp1d

#===============================================================================
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s Progress [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        print("\n")

#==============================================================================
def printDONE(outstr=""):

    print("")
    print("")
    sys.stdout.write("\033[F"); sys.stdout.write("\033[K")
    sys.stdout.write("\r\r[ "+'\033[0;32m'+"DONE "+'\033[0;39m'+"] "+outstr)
    sys.stdout.flush()
    print("")
    print("")

    return

#==============================================================================
def printFAILED(outstr=""):

    print("")
    print("")
    sys.stdout.write("\033[F"); sys.stdout.write("\033[K")
    sys.stdout.write("\r\r[ "+'\033[0;31m'+"FAILED "+'\033[0;39m'+"] "+outstr)
    sys.stdout.flush()
    print("")
    print("")

    return

#==============================================================================
def printWARNING(outstr=""):

    print("")
    print("")
    sys.stdout.write("\033[F"); sys.stdout.write("\033[K")
    sys.stdout.write("\r\r[ "+'\033[0;33m'+"WARNING "+'\033[0;39m'+"] "+outstr)
    sys.stdout.flush()
    print("")
    print("")

    return
    
#==============================================================================
def get_zscores(val):

    mns  = np.nanmean(a=val, keepdims=True)
    sstd = np.nanstd(a=val, keepdims=True)

    return mns, sstd

#==============================================================================
def print_attrs(name,obj):

    print(" - "+name)
    # for key, val in obj.attrs.items():
    #     print("    %s: %s" % (key, val))
 
    return
  
 #============================================================================== 
def show_hdf5_tree(f):
        
    # f = h5py.File(filename,'r')
    f.visititems(print_attrs)
 
    return
