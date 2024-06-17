# -*- coding: utf-8 -*-
import sys

import numpy as np


def interp_weights(xyz, uvw, tri):
    """
    Creates a Delaunay triangulation and finds the vertices and weights of
    points around a given location in parameter space
    """

    d = len(uvw[0, :])
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum("njk,nk->nj", temp[:, :d, :], delta)

    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def printProgress(iteration, total, prefix="", suffix="", decimals=2, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength = int(round(barLength * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = "#" * filledLength + "-" * (barLength - filledLength)
    sys.stdout.write(
        "\r%s Progress [%s] %s%s %s\r" % (prefix, bar, percents, "%", suffix)
    ),
    sys.stdout.flush()
    if iteration == total:
        print("\n")


# ==============================================================================
def get_zscores(val):
    mns = np.nanmean(a=val, keepdims=True)
    sstd = np.nanstd(a=val, keepdims=True)

    return mns, sstd


# ==============================================================================


def print_attrs(name, obj):
    print(" - " + name)
    # for key, val in obj.attrs.items():
    #     print("    %s: %s" % (key, val))

    return


# ==============================================================================


def show_hdf5_tree(f):
    # f = h5py.File(filename,'r')
    f.visititems(print_attrs)

    return
