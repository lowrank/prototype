import numpy as np
import torch

from model import Prototype

from equivariance_check import one_layer, two_layers, check_equiv_weak_formula

from scipy.spatial.transform import Rotation as R


import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


t0 = time.time()

P = Prototype( in_channel=1, out_channel=1, kernel_size=5, \
                 sph_bessel_root=2, sph_harm_index=2, wigner_index=2, so3_grid_type='lebedev', so3_sampling=(26, 8), padding='same', stride=1, standard_layer=False)


Q = Prototype( in_channel=1, out_channel=1, kernel_size=5, \
                 sph_bessel_root=2, sph_harm_index=2, wigner_index=2, so3_grid_type='lebedev',  so3_sampling=(26, 8), padding='same', stride=1, standard_layer=True)

t1 = time.time()

print(t1 - t0)


P = P.to(device)
Q = Q.to(device)

P.check_frame_score()
Q.check_frame_score()


images = torch.zeros(1, 1, 41,41,41, 2, dtype=torch.float32).to(device)

_images_real = np.zeros(images.shape[:-1] ) 
_images_imag = np.zeros(images.shape[:-1] ) 

_images_real[..., 18:25, 18:25, 18:25] = 1.0                                                                     

images[0, 0, :,:,:, 0] = torch.tensor(_images_real)
images[0, 0, :,:,:, 1] = torch.tensor(_images_imag)


for _ in range(10):
    h_index = np.random.randint(low=0, high=P.so3_grid.shape[0])
    g_index = np.random.randint(low=0, high=P.so3_grid.shape[0])

    h_beta, h_alpha,h_gamma = P.so3_grid[h_index]
    g_beta, g_alpha, g_gamma = P.so3_grid[g_index]

    g_rotation = R.from_euler('zyz', [g_gamma ,g_beta,  g_alpha], degrees=False).as_matrix()
    h_rotation = R.from_euler('zyz', [h_gamma ,h_beta,  h_alpha], degrees=False).as_matrix()

    rotation = g_rotation @ h_rotation.transpose()
    rt , tr  = one_layer(P, images, rotation)
    rtt , ttr  = two_layers(P, Q, images, rotation)

    print('trial case one layer: ', _, ' ', torch.norm(tr[:,:, h_index, :, :, :, :].detach() - rt[:, :, g_index, :, :,: , :].detach()) / torch.norm(rt[:, :, g_index, :, :,: , :].detach()))
    print('trial case two layers: ', _, ' ', torch.norm(ttr[:,:, h_index, :, :, :, :].detach() - rtt[:, :, g_index, :, :,: , :].detach()) / torch.norm(rtt[:, :, g_index, :, :,: , :].detach()))
    
    print('checking weak formulation ... ')
    print('one layer: ')
    check_equiv_weak_formula(P, 2, tr, rt, rotation) # check up to order 2.
    print('two layer: ')
    check_equiv_weak_formula(P, 2, ttr, rtt, rotation) # check up to order 2.
    
