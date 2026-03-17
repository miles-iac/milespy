# -*- coding: utf-8 -*-
"""Delaunay-based interpolation weights for parameter-space (e.g. age, metallicity) interpolation."""
import numpy as np


def interp_weights(xyz, uvw, tri):
    """
    Compute vertices and barycentric weights for points in a Delaunay triangulation.

    Given query points `uvw` and a precomputed Delaunay triangulation `tri` of
    points `xyz`, returns the simplex vertices and weights for interpolating
    at the query points.

    Parameters
    ----------
    xyz : array-like
        Not used (tri is precomputed); kept for API compatibility.
    uvw : array-like
        Query points, shape (n_points, n_dim).
    tri : scipy.spatial.Delaunay
        Delaunay triangulation of the parameter space.

    Returns
    -------
    vertices : ndarray
        Indices of the simplex vertices for each query point.
    weights : ndarray
        Barycentric weights (sum to 1) for each vertex.
    """
    d = len(uvw[0, :])
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum("njk,nk->nj", temp[:, :d, :], delta)

    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
