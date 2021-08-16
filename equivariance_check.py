import torch

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates

from utils import wignerD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Rotate Image in space with Euler angles (alpha, beta, gamma)

def rotate_image(images, alpha, beta, gamma):
    # images of shape [N x H X W X D].

    # create meshgrid
    dim     = images.shape

    if len(dim) == 3:
        single_image = True 
    else:
        single_image = False

    axis_x  = np.arange(dim[-3])
    axis_y  = np.arange(dim[-2])
    axis_z  = np.arange(dim[-1])
    
    grid    = np.meshgrid(axis_x, axis_y, axis_z, sparse=False, indexing="ij")

    
    xyz     = np.vstack([grid[0].reshape(-1) - float(dim[-3]-1) / 2,   # x coordinate, centered
                     grid[1].reshape(-1) - float(dim[-2]-1) / 2,       # y coordinate, centered
                     grid[2].reshape(-1) - float(dim[-1]-1) / 2])      # z coordinate, centered

    # create transformation matrix
    mat = R.from_euler('zyz', [gamma, beta, alpha], degrees=False).as_matrix()
    # mat = np.linalg.inv(mat)

    # apply transformation
    transformed_xyz = np.dot(mat, xyz)

    # extract coordinates
    x = transformed_xyz[0, :] + float(dim[-3]-1) / 2
    y = transformed_xyz[1, :] + float(dim[-2]-1) / 2
    z = transformed_xyz[2, :] + float(dim[-1]-1) / 2
    
    new_xyz = np.vstack( (x, y, z) )
 
    # sample
    if not single_image:
        output = np.concatenate([ np.expand_dims(  map_coordinates(images[_, ...], new_xyz, order=1).reshape(dim[-3],dim[-2],dim[-1]), axis=0) for _ in range(dim[0])], axis=0) 
    else:
        output = map_coordinates(images, new_xyz, order=1).reshape(dim[-3],dim[-2],dim[-1])
    
    return output


def rotate_image_mat(images, mat):
    # images of shape [N x H X W X D].

    # create meshgrid
    dim     = images.shape

    if len(dim) == 3:
        single_image = True 
    else:
        single_image = False


    axis_x  = np.arange(dim[-3])
    axis_y  = np.arange(dim[-2])
    axis_z  = np.arange(dim[-1])
    
    grid    = np.meshgrid(axis_x, axis_y, axis_z, sparse=False, indexing="ij")

    
    xyz     = np.vstack([grid[0].reshape(-1) - float(dim[-3]-1) / 2,   # x coordinate, centered
                     grid[1].reshape(-1) - float(dim[-2]-1) / 2,       # y coordinate, centered
                     grid[2].reshape(-1) - float(dim[-1]-1) / 2])      # z coordinate, centered

    # apply transformation
    transformed_xyz = np.dot(mat, xyz)

    # extract coordinates
    x = transformed_xyz[0, :] + float(dim[-3]-1) / 2
    y = transformed_xyz[1, :] + float(dim[-2]-1) / 2
    z = transformed_xyz[2, :] + float(dim[-1]-1) / 2
    
    new_xyz = np.vstack( (x, y, z) )
 
    # sample
    if not single_image:
        output = np.concatenate([ np.expand_dims(  map_coordinates(images[_, ...], new_xyz, order=1).reshape(dim[-3],dim[-2],dim[-1]), axis=0) for _ in range(dim[0])], axis=0) 
    else:
        output = map_coordinates(images, new_xyz, order=1).reshape(dim[-3],dim[-2],dim[-1])
    
    return output



def integral(P, images, l, m , n, rot): 
    if rot is not None:
        alpha_, beta_, gamma_ = convert_to_euler_angles(rot)
    else:
        alpha_, beta_, gamma_ = 0, 0, 0

    ret = np.zeros((images.shape[0], images.shape[1], images.shape[3], images.shape[4], images.shape[5], images.shape[6]), dtype=float)
    for so3_index in range(P.so3_grid.shape[0]):
        beta, alpha, gamma = P.so3_grid[so3_index, :]
        
        Z = R.from_euler('zyz', [gamma, beta,  alpha], degrees=False).as_matrix()
        if rot is not None:
            Q = rot.transpose() @ Z 
        else:
            Q = Z

        a, b, c = convert_to_euler_angles(Q)

        x, y = wignerD(l, m, n, b, a, c)

        # integration on Haar measure.    
        ret[..., 0] = ret[..., 0] + P.so3_weight.cpu()[so3_index].item() * (x  * images[:, :, so3_index, :, :, :, 0].detach().cpu().numpy() - \
                                y * images[:, :, so3_index, :, :, :, 1].detach().cpu().numpy()  ) * (4*np.pi**3/ P.so3_grid.shape[0] )/np.sqrt(8*np.pi**2/(2*l+1))
        ret[..., 1] = ret[..., 1] + P.so3_weight.cpu()[so3_index].item() * (y  * images[:, :, so3_index, :, :, :, 0].detach().cpu().numpy() + \
                                x * images[:, :, so3_index, :, :, :, 1].detach().cpu().numpy()  ) * (4*np.pi**3/ P.so3_grid.shape[0] )/np.sqrt(8*np.pi**2/(2*l+1))

    return ret

def convert_to_euler_angles(mat):
    gamma = np.arctan2(mat[1,2], -mat[0,2])
    alpha = np.arctan2(mat[2,1],  mat[2,0])
    beta = np.arctan2(-mat[0,2] * np.cos(gamma) + mat[1, 2] * np.sin(gamma), mat[2,2])

    return alpha, beta, gamma


def orthogonality_under_rot(rot): 

    alpha_, beta_, gamma_ = convert_to_euler_angles(rot)


    ret = np.zeros((P.g_index.shape[0],P.g_index.shape[0]), dtype=complex)

    for row in range(P.g_index.shape[0]):
        L, j, n = P.g_index[row]
        for col in range(P.g_index.shape[0]):
            L_, j_, n_ = P.g_index[col]

            for so3_index in range(P.so3_grid.shape[0]):
                beta, alpha, gamma = P.so3_grid[so3_index, :]
                
                Z = R.from_euler('zyz', [gamma, beta,  alpha], degrees=False).as_matrix()

                Q = rot.transpose() @ Z 

                a, b, c = convert_to_euler_angles(Q)

                x, y =  wignerD(L, j,n,b,a, c)
                x_, y_= wignerD(L_, j_, n_, b, a, c)

                # integration on Haar measure of (alpha, beta, gamma)
                ret[row, col] += (x + 1j*y) * (x_-1j*y_) * np.sin(beta) * (4*np.pi**3/ P.so3_grid.shape[0] )
   
            ret[row, col] /= np.sqrt( (8*np.pi**2/(2*P.g_index[row][0]+1) ) * (8*np.pi**2/(2*P.g_index[col][0]+1) ) )
            

    return ret



def two_layers(P, Q, images, rotation):

    rot_images_real = rotate_image_mat(images[0, 0, ..., 0].detach().cpu().numpy(), rotation)
    rot_images_imag = rotate_image_mat(images[0, 0, ..., 1].detach().cpu().numpy(), rotation)

    _new_images =  torch.zeros(images.shape, dtype=torch.float32)

    _new_images[0, 0, ..., 0] = torch.tensor( rot_images_real ) # copy 
    _new_images[0, 0, ..., 1] = torch.tensor( rot_images_imag ) # copy 

    _new_images = _new_images.to(device)

    transform_rot_images = P(_new_images)
    transform_transform_rot_images = Q(transform_rot_images)

    transform_images = P(images)
    transform_transform_images = Q(transform_images)

    _rot_transform_transform_images_real = rotate_image_mat(transform_transform_images[0, 0, :, ..., 0].detach().cpu().numpy(), rotation)
    _rot_transform_transform_images_imag = rotate_image_mat(transform_transform_images[0, 0, :, ..., 1].detach().cpu().numpy(), rotation)

    rot_transform_transform_images =  torch.zeros(transform_images.shape, dtype=torch.float32)

    rot_transform_transform_images [0, 0, :, ..., 0] = torch.tensor( _rot_transform_transform_images_real ) # copy 
    rot_transform_transform_images [0, 0, :, ..., 1] = torch.tensor( _rot_transform_transform_images_imag ) # copy 

    rot_transform_transform_images = rot_transform_transform_images.to(device)

    return rot_transform_transform_images, transform_transform_rot_images


def one_layer(P, images, rotation):

    rot_images_real = rotate_image_mat(images[0, 0,:, ..., 0].detach().cpu().numpy(), rotation)
    rot_images_imag = rotate_image_mat(images[0, 0,:, ..., 1].detach().cpu().numpy(), rotation)

    _new_images =  torch.zeros(images.shape, dtype=torch.float32)

    _new_images[0, 0,:, ..., 0] = torch.tensor( rot_images_real ) # copy 
    _new_images[0, 0,:, ..., 1] = torch.tensor( rot_images_imag ) # copy 

    _new_images = _new_images.to(device)

    transform_rot_images = P(_new_images)
    transform_images = P(images)

    _rot_transform_images_real = rotate_image_mat(transform_images[0, 0, :, ..., 0].detach().cpu().numpy(), rotation)
    _rot_transform_images_imag = rotate_image_mat(transform_images[0, 0, :, ..., 1].detach().cpu().numpy(), rotation)

    rot_transform_images =  torch.zeros(transform_images.shape, dtype=torch.float32)

    rot_transform_images [0, 0, :, ..., 0] = torch.tensor( _rot_transform_images_real ) # copy 
    rot_transform_images [0, 0, :, ..., 1] = torch.tensor( _rot_transform_images_imag ) # copy 

    rot_transform_images = rot_transform_images.to(device)

    return rot_transform_images, transform_rot_images


def check_equiv_weak_formula(P, L, tr, rt, rotation):
    # Using Wigner D to check. Largest order L.
    S = 0
    T = 0
    for l in range(L+1):
        for m in range(-l,l+1):
            for n in range(-l,l+1):
                TR = integral(P, tr, l, m, n, None)
                RT = integral(P, rt, l, m, n, rotation)
                S += np.sum((RT-TR)**2)
                T += np.sum(RT**2)
                s = np.sqrt(np.sum((RT-TR)**2))/np.sqrt(np.sum(RT**2))
                t = np.sqrt(np.sum(RT**2))
                print('test case (l, m, n)=(', l, m , n, ')\t, relative error (frobenius) :', s, '\t norm: ', t)
    print('\nL^2 error on Projection Space:', np.sqrt(S/T))

