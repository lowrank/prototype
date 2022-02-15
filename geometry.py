from __future__ import generators


from numpy.ctypeslib import ndpointer
import numpy as np

import ctypes


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
        x_, y_, z_ = zip(*self.points)
        x_min, y_min, z_min = min(x_), min(y_), min(z_)
        x_max, y_max, z_max = max(x_), max(y_), max(z_)
        self.radius = [(x_max - x_min)/2, (y_max - y_min)/2, (z_max - z_min)/2]
        self.center = [(x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2]


def down_sampling(points, weights=None, leaf_size=8):
    # Input:
    #   points     : numpy array of N x D, each row contains the 
    #                coordinates, if weight is not included, then 
    #                all weights are 1. 
    #   weights    : numpy array of N.
    #   leaf_size  : integer, the minimum cluster to be approximated.
    #
    # Output:
    #   down_sampled_point_cloud: numpy array of M x D.

    """
    preprocessing
    """
    diameter_pair = diameter(points)  

    basis = np.zeros((3, 3))

    basis[0] = ( diameter_pair[1] - diameter_pair[0] )

    print(np.linalg.norm(basis[0])     )
    basis[0] = basis[0] / np.linalg.norm(basis[0])         # normalize

    projection = np.eye(3) - np.outer(basis[0], basis[0])  # it is symmetric

    projected_points = points @ projection           

    diameter_pair = diameter(projected_points)

    basis[1]  = ( diameter_pair[1] - diameter_pair[0] )  

    print(np.linalg.norm(basis[1])     )
    basis[1] = basis[1] / np.linalg.norm(basis[1])         # normalize

    basis[2] = np.cross(basis[0], basis[1])                # right-hand-rule.

    _x = points @ basis[0]
    _y = points @ basis[1]
    _z = points @ basis[2]

    # Rotate the points
    points = np.vstack((_x, _y, _z)).T

    """
    down sampling
    """
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
                        np.array(neighbor_weights[neighbor_id]))
                    
                    queue.append(cur_node.children[neighbor_id])
        else:
            # it is a leaf node 
            down_sampled_points.append (
                np.sum(cur_node.points * 
                       cur_node.weights[:, None] 
                       / np.sum (cur_node.weights), 
                       axis = 0 ))
            down_sampled_weights.append(np.sum (cur_node.weights))

    return np.array(down_sampled_points @ basis), np.array(down_sampled_weights)

def _orientation(p,q,r):
    return (q[1]-p[1])*(r[0]-p[0]) - (q[0]-p[0])*(r[1]-p[1])

def _hulls(Points):
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


