import torch


from torch.nn import ReLU
import torch.nn.functional as F


import numpy as np

import matplotlib.pyplot as plt

from utils import generate_so3_lebedev, generate_so3_sampling_grid, spherical_bessel_roots, spherical_bessel_basis, cartesian_spherical, spherical_harmonics, wignerD

class Prototype(torch.nn.Module):
    def __init__(self, in_channel=1, out_channel=1, \
                 kernel_size=5, sph_bessel_root=3, sph_harm_index=3, wigner_index=3, so3_grid_type='lebedev', so3_sampling=(26, 8), \
                 padding='same', stride=1,\
                 standard_layer=True):
        
        super(Prototype, self).__init__()

        self.in_channel       = in_channel
        self.out_channel      = out_channel 
        self.kernel_size      = (kernel_size - 1)//2

        self.sph_bessel_root  = sph_bessel_root
        self.sph_harm_index   = sph_harm_index
        self.wigner_index     = wigner_index
        self.so3_sampling     = so3_sampling

        self.padding          = padding
        self.stride           = stride

        self.standard_layer   = standard_layer

        _root_index       = np.arange(1, self.sph_bessel_root + 1)   
        _harm_index       = np.concatenate( [ np.arange(-index, index  + 1) for index in range(self.sph_harm_index + 1)]  )

        _harm_degree      = np.concatenate( [ torch.tensor( [index] * (2 * index + 1 ) ) for index in range(self.sph_harm_index + 1)] * self.sph_bessel_root )
        
        _root_grid, _sph_grid  = np.meshgrid(_root_index, _harm_index, sparse=False, indexing="ij") 
        _root_grid, _sph_grid  = _root_grid.flatten(), _sph_grid.flatten()

        self.v_index          = np.stack((_root_grid, _harm_degree, _sph_grid), axis=1)    

        _wiger_degree         = np.concatenate( [ [index] * (2 * index + 1 )**2 for index in range(self.wigner_index + 1)] )

        _wiger_orders         = np.array([], dtype=np.int32).reshape(0, 2)    

        # it seems swapping _j, _n is better in later computation when cutting the tensor.
       
        for index in range(self.wigner_index + 1):
            _j, _n = np.meshgrid( np.arange(-index, index  + 1) , np.arange(-index, index  + 1), sparse=False, indexing="ij") 
            _j, _n = _j.flatten(), _n.flatten()

            _local_index = np.stack((_j, _n), axis=1)                                                                   # (2 * index + 1)**2 x 2. 
            _wiger_orders = np.row_stack( (_wiger_orders, _local_index) ) 

        self.g_index = np.column_stack( (_wiger_degree, _wiger_orders))                                                 

        roots  = spherical_bessel_roots(self.sph_harm_index, self.sph_bessel_root) 

        if so3_grid_type == 'lebedev':
            so3_grid, so3_weight = generate_so3_lebedev(self.so3_sampling[0], self.so3_sampling[1])
        elif so3_grid_type == 'uniform':
            so3_grid, so3_weight = generate_so3_sampling_grid(self.so3_sampling[0], self.so3_sampling[1], self.so3_sampling[2])
        
        self.so3_grid = so3_grid


        if self.sph_harm_index <= self.wigner_index:
            # this is stored on GPU with torch tensor.
            cache = torch.zeros( ( self.g_index.shape[0], self.so3_grid.shape[0], 2) , dtype=torch.float32)

            for g_order, g_idx in enumerate(self.g_index):
                for so3_idx in range(self.so3_grid.shape[0]):
                    beta, alpha, gamma = self.so3_grid[so3_idx, :]
                    cache[g_order, so3_idx, 0], cache[g_order, so3_idx, 1] =  wignerD(g_idx[0], g_idx[1], g_idx[2], beta, alpha, gamma) 
        else:
            _cache_degree         = np.concatenate( [ [index] * (2 * index + 1 )**2 for index in range(self.sph_harm_index + 1)] )
            _cache_orders         = np.array([], dtype=np.int32).reshape(0, 2)    

            for index in range(self.sph_harm_index + 1):
                _j, _n = np.meshgrid( np.arange(-index, index  + 1) , np.arange(-index, index  + 1), sparse=False, indexing="ij") 
                _j, _n = _j.flatten(), _n.flatten()

                _local_index = np.stack((_j, _n), axis=1)                                                                 
                _cache_orders = np.row_stack( (_cache_orders, _local_index) ) 

            _cache_index = np.column_stack( (_cache_degree, _cache_orders))
                    
            cache = torch.zeros( ( _cache_index.shape[0], self.so3_grid.shape[0], 2) , dtype=torch.float32)

            for g_order, g_idx in enumerate(_cache_index):
                for so3_idx in range(self.so3_grid.shape[0]):
                    beta, alpha, gamma = self.so3_grid[so3_idx, :]
                    cache[g_order, so3_idx, 0], cache[g_order, so3_idx, 1] =  wignerD(g_idx[0], g_idx[1], g_idx[2], beta, alpha, gamma)             



        # Kernels with respect to spherical harmonics.
        kernel_range = np.arange(-self.kernel_size, self.kernel_size + 1)
        _x, _y, _z   = np.meshgrid(kernel_range, kernel_range, kernel_range, sparse=False, indexing="ij")
        _x, _y, _z   = _x.flatten(), _y.flatten(), _z.flatten()  
        kernel_grid  = np.stack((_x, _y, _z), axis=1)                                                                   # {(2 * kernel + 1)^3 } x 3.
        kernel_grid  = kernel_grid / self.kernel_size      

        theta, phi, r = cartesian_spherical(kernel_grid[:, 0], kernel_grid[:, 1], kernel_grid[:, 2])

        mask   = r <= 1

        kernel_real = torch.zeros( (self.v_index.shape[0], 2 * self.kernel_size + 1 , 2 * self.kernel_size + 1  , 2 * self.kernel_size + 1), dtype=torch.float32)
        kernel_imag = torch.zeros( (self.v_index.shape[0], 2 * self.kernel_size + 1 , 2 * self.kernel_size + 1  , 2 * self.kernel_size + 1), dtype=torch.float32)

        # Construct kernel, this is inefficient.
        for v_order, v_idx in enumerate(self.v_index):
            q, l, m = v_idx[0], v_idx[1], v_idx[2]

            # kernel_values shape: { (2 * kernel + 1)^3 } complex valued. 
            # q starts from 1.
            kernel_values = mask * spherical_bessel_basis(l, roots[l, q-1], r) * np.conj( spherical_harmonics(m, l, theta, phi) )  / np.sqrt(2 * l + 1) / ((2 * self.kernel_size + 1)**3 ) 
            kernel_values = torch.tensor(kernel_values, dtype=torch.complex64).view(2 * self.kernel_size + 1,  2 * self.kernel_size + 1,  2 * self.kernel_size + 1)

            # ===> {Q x (L+1)^2 }  x (2K+1) x (2K+1) x (2K+1)  x 2
            # Typical Data Size: 3 x 16 x 11^3 x 2 x (float32) = 0.48 MBytes.
            kernel_real[v_order, ...] = kernel_values.real
            kernel_imag[v_order, ...] = kernel_values.imag



        # Construct tensor T (real/imag) and W(real/imag). 

        T_real = torch.zeros(  (self.v_index.shape[0], self.v_index.shape[0], self.so3_grid.shape[0]), dtype=torch.float32) 
        T_imag = torch.zeros(  (self.v_index.shape[0], self.v_index.shape[0], self.so3_grid.shape[0]), dtype=torch.float32) 

        if self.standard_layer:
            # 1st layer does not need W.
            W_real = torch.zeros(  (self.g_index.shape[0], self.g_index.shape[0], self.so3_grid.shape[0]), dtype=torch.float32) 
            W_imag = torch.zeros(  (self.g_index.shape[0], self.g_index.shape[0], self.so3_grid.shape[0]), dtype=torch.float32) 

        for so3_index in range(self.so3_grid.shape[0]):
            for v_order_m, v_idx_m in enumerate(self.v_index):
                q, l, m = v_idx_m[0], v_idx_m[1], v_idx_m[2]
                for v_order_s, v_idx_s in enumerate(self.v_index):
                    q_, l_, s = v_idx_s[0], v_idx_s[1], v_idx_s[2]
                    if q == q_ and l == l_:
                        # conj ( D_l^{s,m} (h) )
                        T_order = ( l * (2 * l - 1) * (2 * l + 1) )//3 + (s + l) * (2 * l + 1) + (m + l)
                        T_real[v_order_m, v_order_s, so3_index] =  cache[T_order, so3_index, 0]
                        T_imag[v_order_m, v_order_s, so3_index] =  -cache[T_order, so3_index, 1] # conj
                    
        # 1st layer does not need W.
        if self.standard_layer:
            for so3_index in range(self.so3_grid.shape[0]):
                for g_order_j, g_idx_j in enumerate(self.g_index):
                    b, j, n = g_idx_j[0], g_idx_j[1], g_idx_j[2]
                    for g_order_t, g_idx_t in enumerate(self.g_index):
                        b_, t, n_ = g_idx_t[0], g_idx_t[1], g_idx_t[2]
                        if b == b_ and n == n_:# check later
                            W_order = ( b * (2 * b - 1) * (2 * b + 1) )//3 + (t + b) * (2 * b + 1) + (j + b)
                            
                            W_real[g_order_j, g_order_t, so3_index] = cache[W_order, so3_index, 0]
                            W_imag[g_order_j, g_order_t, so3_index] = cache[W_order, so3_index, 1]
                    
        
        self.register_buffer('so3_weight',  so3_weight)
        self.register_buffer('cache', cache)
        self.register_buffer('kernel_real', kernel_real)
        self.register_buffer('kernel_imag', kernel_imag)
        
        if self.standard_layer:
            self.register_buffer('W_real', W_real)
            self.register_buffer('W_imag', W_imag)
            
        self.register_buffer('T_real', T_real)
        self.register_buffer('T_imag', T_imag)

        # set parameter for this layer. Requires grad.

        init_param   = torch.rand( self.in_channel, self.out_channel,  self.g_index.shape[0], self.v_index.shape[0])

        self.weights =  torch.nn.Parameter(init_param , requires_grad=True)

    def compute_G(self, batch_images):
        # compute the G functions from batch_images of size:
        #   standard_layer:   {Batch Size (b)} x {Channel In (i)} x {Group Size (g)} x {H} x {W} x {D} x {real, imag}
        #   otheriwse:        {Batch Size (b)} x {Channel In (i)} x {H} x {W} x {D} x {real, imag}
        # 
        # Typical Input Batch Image Data 4 x 8 x 512 x 32 x 32 x 32 x 2 (float32) = 4 GByte.
        # GPU computation would be difficult if all ops are on GPU. 

        # compute H functions through SO3 integral by einsum on {Group Size}. 
        #
        # Output is 
        #   standard_layer: {Batch Size} x {Channel In} x {g_index} x {H} x {W} x {D} (x 2)
        #   otherwise:      {Batch Size} x {Channel In}             x {H} x {W} x {D} (x 2)
        # Typical Output Batch Data 2 x 4 x 84 x 32 x 32 x 32 (x 2) (float32) = 0.165 GByte.
        # Remember the conjugate (negative sign)
        # Should use GPU (by loading the data on GPU).

        H, W, D = batch_images.shape[-4], batch_images.shape[-3], batch_images.shape[-2] 

        if self.standard_layer:

            H_real =  torch.einsum('bighwd, kg->bikhwd', batch_images[..., 0],  ( self.cache[:self.g_index.shape[0], :, 0] ) * self.so3_weight.view(1,-1)) + \
            torch.einsum('bighwd, kg->bikhwd', batch_images[..., 1],  ( self.cache[:self.g_index.shape[0], :,1] ) * self.so3_weight.view(1,-1))

            H_imag = -torch.einsum('bighwd, kg->bikhwd', batch_images[..., 0],  ( self.cache[:self.g_index.shape[0], :, 1] ) * self.so3_weight.view(1,-1)) + \
            torch.einsum('bighwd, kg->bikhwd', batch_images[..., 1],  ( self.cache[:self.g_index.shape[0], :, 0] ) * self.so3_weight.view(1,-1))
        
        else:
            H_real =  batch_images[..., 0]

            H_imag =  batch_images[..., 1]

        # Convolution with Kernels (real/imag parts). The convolution takes 4 real convolutions since torch only supports real ops.

        # Batch Image:
        #   standard_layer: [ {Batch Size} x {Channel In} x {g_index} ] x {H} x {W} x {D}
        #   otherwise:      [ {Batch Size} x {Channel In}  ]            x {H} x {W} x {D}

        # Conv3D Kernels: {Q x (L+1)^2 } x 1 x (2K+1) x (2K+1) x (2K+1)

        # Out: 
        #   standard_layer: {Batch Size} x {Channel In} x {g_index} x {v_index} x {H} x {W} x {D}
        #   otherwise:      {Batch Size} x {Channel In} x             {v_index} x {H} x {W} x {D}
        #
        # Data Size: 2 x 4 x 84 x (3 x 35) x 32 x 32 x 32 (x 2) float ~ 6 GBytes?

        real_part = torch.nn.functional.conv3d(input = H_real.reshape(-1, 1, H, W, D ) , \
                                weight = self.kernel_real.view(-1, 1, 2 * self.kernel_size + 1 , 2 * self.kernel_size + 1 , 2 * self.kernel_size + 1 ), \
                                padding = self.padding, stride=self.stride)
        
        real_part -=  torch.nn.functional.conv3d(input = H_imag.reshape(-1, 1, H, W, D) , \
                                weight = self.kernel_imag.view(-1, 1, 2 * self.kernel_size + 1 , 2 * self.kernel_size + 1 , 2 * self.kernel_size + 1 ), \
                                padding = self.padding, stride=self.stride)
        
        imag_part = torch.nn.functional.conv3d(input = H_real.reshape(-1, 1, H, W, D) , \
                                weight = self.kernel_imag.view(-1, 1, 2 * self.kernel_size + 1 , 2 * self.kernel_size + 1 , 2 * self.kernel_size + 1 ), \
                                padding = self.padding, stride=self.stride)
        
        imag_part +=  torch.nn.functional.conv3d(input = H_imag.reshape(-1, 1, H, W, D) , \
                                weight = self.kernel_real.view(-1, 1, 2 * self.kernel_size + 1 , 2 * self.kernel_size + 1 , 2 * self.kernel_size + 1 ), \
                                padding = self.padding, stride=self.stride)
        
        
        if self.standard_layer:
            # without g_index.
            return  real_part.view(batch_images.shape[0], \
                                batch_images.shape[1], \
                                self.g_index.shape[0], \
                                self.v_index.shape[0], \
                                real_part.shape[-3], \
                                real_part.shape[-2], \
                                real_part.shape[-1]), \
                    imag_part.view(batch_images.shape[0], \
                                batch_images.shape[1], \
                                self.g_index.shape[0], \
                                self.v_index.shape[0], \
                                imag_part.shape[-3], \
                                imag_part.shape[-2], \
                                imag_part.shape[-1])
                    
        else:
            # non standard layer
            return  real_part.view(batch_images.shape[0], \
                                batch_images.shape[1], \
                                self.v_index.shape[0], \
                                real_part.shape[-3], \
                                real_part.shape[-2], \
                                real_part.shape[-1]), \
                    imag_part.view(batch_images.shape[0], \
                                batch_images.shape[1], \
                                self.v_index.shape[0], \
                                imag_part.shape[-3], \
                                imag_part.shape[-2], \
                                imag_part.shape[-1])       
        

    def non_standard_forward(self, batch_images):

        # both shape:
        #   standard: {Batch Size} x {Channel In} x {g_index} x {v_index} x {H'} x {W'} x {D'} (depends on convolution arguments).
        #   otherwise: {Batch Size} x {Channel In}            x {v_index} x {H'} x {W'} x {D'} (depends on convolution arguments).

        G_real, G_imag = self.compute_G(batch_images) 

        # parameters. {Channel In} x {Channel Out} x {g_index} x {v_index}

        # set output: {Batch Size} x {Channel Out} x {Group Size} x {H} x {W} x {D} x {real, imag}

        # There are 8 einsum to compute...

        # real, rr

        einsum_str1 = 'iokv, vVg->iokVg'
        einsum_str2 = 'iokVg, biVhwd->boghwd'

        output = torch.einsum(einsum_str1,         self.weights, self.T_real)
        output_real = torch.einsum(einsum_str2,    output, G_real)

        output = torch.einsum(einsum_str1,          self.weights, self.T_imag)
        output_ii = torch.einsum(einsum_str2,      output, G_imag)

        output_real -= output_ii

        output_real = output_real.unsqueeze(dim=-1)

        # imag ri
        output = torch.einsum(einsum_str1,          self.weights, self.T_real)
        output_imag = torch.einsum(einsum_str2,     output, G_imag)

        output = torch.einsum(einsum_str1,          self.weights, self.T_imag)
        output_ir = torch.einsum(einsum_str2,       output, G_real)

        output_imag += output_ir

        output_imag = output_imag.unsqueeze(dim=-1)

        return torch.cat((output_real, output_imag), dim=-1) # Batch x Out Channel x Group x H x W x D


    def standard_forward(self, batch_images):
        

        # both shape:
        #   standard: {Batch Size} x {Channel In} x {g_index} x {v_index} x {H'} x {W'} x {D'} (depends on convolution arguments).
        #   otherwise: {Batch Size} x {Channel In}            x {v_index} x {H'} x {W'} x {D'} (depends on convolution arguments).

        G_real, G_imag = self.compute_G(batch_images) 

        # parameters. {Channel In} x {Channel Out} x {g_index} x {v_index}

        # set output: {Batch Size} x {Channel Out} x {Group Size} x {H} x {W} x {D} x {real, imag}

        # There are 8 einsum to compute...

        # real
        output = torch.einsum('iokv, kKg->ioKvg',          self.weights, self.W_real)
        output = torch.einsum('ioKvg, vVg->ioKVg',               output, self.T_real)
        output_real = torch.einsum('ioKVg, biKVhwd->boghwd',     output, G_real)


        output = torch.einsum('iokv, kKg->ioKvg',          self.weights, self.W_imag)
        output = torch.einsum('ioKvg, vVg->ioKVg',               output, self.T_imag)
        output_iir = torch.einsum('ioKVg, biKVhwd->boghwd',      output, G_real)

        output_real -= output_iir 

        output = torch.einsum('iokv, kKg->ioKvg',          self.weights, self.W_imag)
        output = torch.einsum('ioKvg, vVg->ioKVg',               output, self.T_real)
        output_iri = torch.einsum('ioKVg, biKVhwd->boghwd',      output, G_imag)

        output_real -= output_iri 

        output = torch.einsum('iokv, kKg->ioKvg',          self.weights, self.W_real)
        output = torch.einsum('ioKvg, vVg->ioKVg',               output, self.T_imag)
        output_rii = torch.einsum('ioKVg, biKVhwd->boghwd',      output, G_imag)

        output_real -= output_rii

        output_real = output_real.unsqueeze(dim=-1)

        # imag
        output = torch.einsum('iokv, kKg->ioKvg',          self.weights, self.W_real)
        output = torch.einsum('ioKvg, vVg->ioKVg',               output, self.T_real)
        output_imag = torch.einsum('ioKVg, biKVhwd->boghwd',      output, G_imag)

        output = torch.einsum('iokv, kKg->ioKvg',          self.weights, self.W_real)
        output = torch.einsum('ioKvg, vVg->ioKVg',               output, self.T_imag)
        output_rir = torch.einsum('ioKVg, biKVhwd->boghwd',      output, G_real)

        output_imag += output_rir

        output = torch.einsum('iokv, kKg->ioKvg',          self.weights, self.W_imag)
        output = torch.einsum('ioKvg, vVg->ioKVg',               output, self.T_real)
        output_irr = torch.einsum('ioKVg, biKVhwd->boghwd',      output, G_real)

        output_imag += output_irr

        output = torch.einsum('iokv, kKg->ioKvg',          self.weights, self.W_imag)
        output = torch.einsum('ioKvg, vVg->ioKVg',               output, self.T_imag)
        output_iii = torch.einsum('ioKVg, biKVhwd->boghwd',      output, G_imag)

        output_imag -= output_iii

        output_imag = output_imag.unsqueeze(dim=-1)

        return torch.cat((output_real, output_imag), dim=-1) # Batch x Out Channel x Group x H x W x D


    def forward(self, batch_images):
        if self.standard_layer:
            return self.standard_forward(batch_images)
        else:
            return self.non_standard_forward(batch_images)
        
        
        
        

    def check_frame_score(self):
        local_mesh_mat = np.zeros((self.v_index.shape[0], self.v_index.shape[0] ), dtype=complex )
        for row in range(self.v_index.shape[0]):
            for col in range(self.v_index.shape[0]):
                real_val =  torch.sum(self.kernel_real[row, :, :, :] * self.kernel_real[col, :, :, :] + self.kernel_imag[row, :, :, :] * self.kernel_imag[col, :, :, :]).item()
                imag_val =  torch.sum(-self.kernel_real[row, :, :, :] * self.kernel_imag[col, :, :, :] +  self.kernel_imag[row, :, :, :] * self.kernel_real[col, :, :, :]).item()
                local_mesh_mat[row, col] = real_val + 1j * imag_val 

        # normalize.
        for row in range(self.v_index.shape[0]):
            for col in range(self.v_index.shape[0]):
                local_mesh_mat[row, col] =  local_mesh_mat[row, col] / np.sqrt(local_mesh_mat[row, row].real   * local_mesh_mat[col, col].real)

        svd_vals = np.linalg.svd(local_mesh_mat, compute_uv=False)

        print( 'local 3d mesh score: (largest gap between 1 and singular values)', np.max( [np.abs(1-np.min(svd_vals)) , np.abs(1-np.max(svd_vals))] ) )


        local_wigner_mat = np.zeros((self.g_index.shape[0], self.g_index.shape[0]), dtype=complex)

        for row in range(self.g_index.shape[0]):
            for col in range(self.g_index.shape[0]):
                real_part = torch.sum( ( self.cache[row, :, 0] * self.cache[col, :, 0] +  self.cache[row, :, 1] * self.cache[col, :, 1] ) * self.so3_weight ).item()  * (4*np.pi**3/ self.so3_grid.shape[0] )
                imag_part = torch.sum( (- self.cache[row, :, 0] * self.cache[col, :, 1] +  self.cache[row, :, 1] * self.cache[col, :, 0] ) * self.so3_weight ).item() * (4*np.pi**3/ self.so3_grid.shape[0] )

                local_wigner_mat[row, col] = real_part + 1j * imag_part

        
        # normalize.
        for row in range(self.g_index.shape[0]):
            for col in range(self.g_index.shape[0]):
                local_wigner_mat[row, col] =  local_wigner_mat[row, col] / np.sqrt(local_wigner_mat[row, row].real   * local_wigner_mat[col, col].real)


        wigner_svd_vals = np.linalg.svd(local_wigner_mat, compute_uv=False)

        print( 'group frame score: (largest gap between 1 and singular values)', np.max( [np.abs(1-np.min(wigner_svd_vals)) , np.abs(1-np.max(wigner_svd_vals))] ) )
