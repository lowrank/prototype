from __future__ import generators

import numpy as np

import ctypes
from numpy.ctypeslib import ndpointer

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class _node(object):
    def __init__(self, points, weights):
        self.points = points
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(self.points.shape[0])

        self.get_information()
        self.children = [None] * 8

    def get_information(self):
        # todo: get the furthest two points as x-axis.
        #       get the other two axis through mass center, orthogonal to x-axis.
        #       project all points onto yz-plane, locate the furthest two points as y-axis.
        #       z-axis is then determined.

        x_, y_, z_= zip(*self.points)
        x_min, y_min, z_min = min(x_), min(y_), min(z_)
        x_max, y_max, z_max = max(x_), max(y_), max(z_)
        self.radius = [(x_max - x_min)/2, (y_max - y_min)/2, (z_max - z_min)/2]
        self.center = [(x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2]

def down_sampling(points, weights=None, leaf_size=8):
    # Input:
    #   point_cloud: numpy array of N x (D + 1) or N x D, each row contains the 
    #                coordinates and (weight), if weight is not included, then 
    #                all weights are 1. 
    #   leaf_size  : integer, the minimum cluster to be approximated.
    #
    # Output:
    #   down_sampled_point_cloud: numpy array of M x D.
    
    root = _node(points, weights)
    
    down_sampled_points = []
    down_sampled_weights = []
    
    queue = []
    queue.append(root)

    while queue:
        cur_node = queue.pop(0)
        if cur_node.points.shape[0] > leaf_size:

            neighbor_points = [[], [], [], [], [], [], [], []] 
            neighbor_weights = [[], [], [], [], [], [], [], []] 

            for id in range(cur_node.points.shape[0]):
                pos = cur_node.points[id] > cur_node.center
                neighbor_id = 4 * pos[0] + 2 * pos[1] + pos[2]
                neighbor_points[neighbor_id].append( cur_node.points[id] )
                neighbor_weights[neighbor_id].append( cur_node.weights[id] )

            for neighbor_id in range(8):
                if neighbor_points[neighbor_id]:

                    cur_node.children[neighbor_id] = _node( 
                        np.array(neighbor_points[neighbor_id]), 
                        np.array(neighbor_weights[neighbor_id]) 
                        )
                    
                    queue.append(cur_node.children[neighbor_id])
        else:
            # it is a leaf node 
            down_sampled_points.append (
                np.sum(cur_node.points * 
                       cur_node.weights[:, None] 
                       / np.sum (cur_node.weights), 
                       axis = 0 ))
            down_sampled_weights.append(np.sum (cur_node.weights))

    return np.array(down_sampled_points), np.array(down_sampled_weights)

def _orientation(p,q,r):
    '''Return positive if p-q-r are clockwise, neg if ccw, zero if colinear.'''
    return (q[1]-p[1])*(r[0]-p[0]) - (q[0]-p[0])*(r[1]-p[1])

def _hulls(Points):
    '''Graham scan to find upper and lower convex hulls of a set of 2d points.'''
    U = []
    L = []
    Points.sort()
    for p in Points:
        while len(U) > 1 and _orientation(U[-2],U[-1],p) <= 0: U.pop()
        while len(L) > 1 and _orientation(L[-2],L[-1],p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U,L

def _rotatingCalipers(Points):
    '''Given a list of 2d points, finds all ways of sandwiching the points
between two parallel lines that touch one point each, and yields the sequence
of pairs of points touched by each pair of lines.'''
    U,L = _hulls(Points)
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i],L[j]
        
        # if all the way through one side of hull, advance the other side
        if i == len(U) - 1: j -= 1
        elif j == 0: i += 1
        
        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif (U[i+1][1]-U[i][1])*(L[j][0]-L[j-1][0]) > \
                (L[j][1]-L[j-1][1])*(U[i+1][0]-U[i][0]):
            i += 1
        else: j -= 1

def diameter(Points):
    '''Given a list of 2d points, returns the pair that's farthest apart.'''
    if Points.shape[1] == 2:
        _, pair = max([((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
                        for p,q in _rotatingCalipers(Points)])
        return pair
    elif Points.shape[1] == 3:
        p = np.zeros(3)
        q = np.zeros(3)

        n = Points.shape[0]
        points = Points.flatten("C") # row major
        diameter_wrapper = ctypes.CDLL("./libgdiam.so")._Z30gdiam_approx_diam_pair_wrapperPdiS_S_ 

        diameter_wrapper.restype = None
        diameter_wrapper.argtypes = [ndpointer(ctypes.c_double), 
                                    ctypes.c_int, 
                                    ndpointer(ctypes.c_double), 
                                    ndpointer(ctypes.c_double)]

        diameter_wrapper(points, n, p, q)

        return p, q


