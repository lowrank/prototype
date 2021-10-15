import numpy as np
import torch

from model import Prototype

from equivariance_check import check_equivariance,  check_equiv_weak_formula
from scipy.spatial.transform import Rotation as R


import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


t0 = time.time()

P = Prototype( in_channel=1, out_channel=1, kernel_size=5, \
                 sph_bessel_root=2, sph_harm_index=2, wigner_index=1, so3_sampling=[(38, 8), (38, 8)], padding='same', stride=1, standard_layer=False)


Q = Prototype( in_channel=1, out_channel=1, kernel_size=5, \
                 sph_bessel_root=2, sph_harm_index=2, wigner_index=1, so3_sampling=[(38, 8), (38, 8)], padding='same', stride=1, standard_layer=True)

t1 = time.time()

print(t1 - t0)


P = P.to(device)
Q = Q.to(device)

P.check_frame_score()
Q.check_frame_score()


images = torch.zeros(1, 1, 41,41,41, 2, dtype=torch.float32).to(device)

_images_real = np.zeros(images.shape[:-1] ) 
_images_imag = np.zeros(images.shape[:-1] ) 

_images_real[..., 10:30, 10:30, 10:30] = 1.0          
# _images_real[..., 18:20, 18:20, 18:20] = 0.0                                                             

images[0, 0, :,:,:, 0] = torch.tensor(_images_real)
images[0, 0, :,:,:, 1] = torch.tensor(_images_imag)

f = lambda x: Q(P(x))
g = lambda x: P(x)

res = 0
for _ in range(50):
    h_index = np.random.randint(low=0, high=Q.output_so3_grid.shape[0] )
    g_index = np.random.randint(low=0, high=Q.output_so3_grid.shape[0] )

    h_beta, h_alpha,h_gamma  = Q.output_so3_grid[h_index]
    g_beta, g_alpha, g_gamma = Q.output_so3_grid[g_index]

    g_rotation = R.from_euler('zyz', [g_gamma ,g_beta,  g_alpha], degrees=False).as_matrix()
    h_rotation = R.from_euler('zyz', [h_gamma ,h_beta,  h_alpha], degrees=False).as_matrix()

    rotation = g_rotation @ h_rotation.transpose()
    rt , tr  = check_equivariance(f, images, rotation)

    acc = torch.norm(tr[:,:, h_index, :, :, :, :].detach() - rt[:, :, g_index, :, :,: , :].detach()) / torch.norm(tr[:, :, h_index, :, :,: , :].detach())


    # check_equiv_weak_formula(Q.output_so3_grid, Q.output_so3_weight.cpu().numpy(), 1, tr, rt, rotation)
    res += acc

    print('check error: ', _, ' ',  acc, res / (_+1))
