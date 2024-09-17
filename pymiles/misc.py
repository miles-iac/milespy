# -*- coding: utf-8 -*-
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
