import torch
import torch.nn.functional as F
import numpy as np

class Prototype(torch.nn.Module):
    def __init__(self, in_channel=1, out_channel=1, \
                 kernel_size=5, sph_bessel_root=3, sph_harm_index=3, wigner_index=3, \
                 so3_sampling=[(26, 8), (14, 8)], \
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

        _wiger_degree         = np.concatenate( [ [index] * (2 * index + 1 )**2  for index in range(self.wigner_index + 1)] )  # number of basis changed.

        _wiger_orders         = np.array([], dtype=np.int32).reshape(0, 2)    
       
        for index in range(self.wigner_index + 1):           

            _j, _n = np.meshgrid( np.arange(-index, index  + 1), np.arange(-index, index  + 1), sparse=False, indexing="ij") 
            _j, _n = _j.flatten(), _n.flatten()

            _local_index = np.stack((_j, _n), axis=1)                                                                   # (2 * index + 1)**2 x 2. 
            _wiger_orders = np.row_stack( (_wiger_orders, _local_index) ) 

        self.g_index = np.column_stack( (_wiger_degree, _wiger_orders) )                                                 

        roots  = spherical_bessel_roots(self.sph_harm_index, self.sph_bessel_root) 

        # input/output grid
        input_so3_grid,  input_so3_weight   = generate_so3_lebedev(self.so3_sampling[0][0],  self.so3_sampling[0][1])
        output_so3_grid, output_so3_weight  = generate_so3_lebedev(self.so3_sampling[1][0],  self.so3_sampling[1][1])

        self.input_so3_grid  = input_so3_grid
        self.output_so3_grid = output_so3_grid

        # generate cache

        """
        @update: cache will drop the last index.
        
        The (input) cache is used for input SO3 grid and output_cache is used at output SO3 grid.
        """

        input_wigner_basis  = torch.zeros( ( self.g_index.shape[0], self.input_so3_grid.shape[0]) , dtype=torch.float32)

        for g_order, g_idx in enumerate(self.g_index):
            for so3_idx in range(self.input_so3_grid.shape[0]):
                beta, alpha, gamma = self.input_so3_grid[so3_idx, :]

                input_wigner_basis[g_order, so3_idx] =  wignerD(g_idx[0], g_idx[1], g_idx[2], beta, alpha, gamma) 

        if self.sph_harm_index <= self.wigner_index:
            # this is stored on GPU with torch tensor.

            
            output_wigner_basis = torch.zeros( ( self.g_index.shape[0], self.output_so3_grid.shape[0]) , dtype=torch.float32)

            for g_order, g_idx in enumerate(self.g_index):
                for so3_idx in range(self.output_so3_grid.shape[0]):
                    beta, alpha, gamma = self.output_so3_grid[so3_idx, :]
                    output_wigner_basis[g_order, so3_idx] =  wignerD(g_idx[0], g_idx[1], g_idx[2], beta, alpha, gamma) 

        else:
            _tmp_degree         = np.concatenate( [ [index] * (2 * index + 1)**2 for index in range(self.sph_harm_index + 1)] )
            _tmp_orders         = np.array([], dtype=np.int32).reshape(0, 2)    

            for index in range(self.sph_harm_index + 1): 

                _j, _n = np.meshgrid(np.arange(-index, index  + 1) ,  np.arange(-index, index  + 1), sparse=False, indexing="ij") 
                _j, _n = _j.flatten(), _n.flatten()

                _tmp_index = np.stack((_j, _n), axis=1)                                                                 
                _tmp_orders = np.row_stack( (_tmp_orders, _tmp_index) ) 

            _tmp_index = np.column_stack( (_tmp_degree, _tmp_orders))
                    
            output_wigner_basis = torch.zeros( ( _tmp_index.shape[0], self.output_so3_grid.shape[0]) , dtype=torch.float32)

            for g_order, g_idx in enumerate(_tmp_index):
                for so3_idx in range(self.output_so3_grid.shape[0]):
                    beta, alpha, gamma = self.output_so3_grid[so3_idx, :]
                    output_wigner_basis[g_order, so3_idx]=  wignerD(g_idx[0], g_idx[1], g_idx[2], beta, alpha, gamma)    


        # filters with respect to spherical harmonics.
        kernel_range = np.arange(-self.kernel_size, self.kernel_size + 1)
        _x, _y, _z   = np.meshgrid(kernel_range, kernel_range, kernel_range, sparse=False, indexing="ij")
        _x, _y, _z   = _x.flatten(), _y.flatten(), _z.flatten()  
        kernel_grid  = np.stack((_x, _y, _z), axis=1)                                                                   # {(2 * kernel + 1)^3 } x 3.
        kernel_grid  = kernel_grid / (self.kernel_size + 1)                                                             # todo: weights to be reformulated.

        theta, phi, r = cartesian_spherical(kernel_grid[:, 0], kernel_grid[:, 1], kernel_grid[:, 2])

        mask   = r <= 1

        kernel = torch.zeros( (self.v_index.shape[0], 2 * self.kernel_size + 1 , 2 * self.kernel_size + 1  , 2 * self.kernel_size + 1), dtype=torch.float32)
 
        # Construct kernel, this is inefficient.
        for v_order, v_idx in enumerate(self.v_index):
            q, l, m = v_idx[0], v_idx[1], v_idx[2]

            # kernel_values shape: { (2 * kernel + 1)^3 } complex valued. 
            # q starts from 1.
            """
            @update: spherical harmonic is real now.
            """
            kernel_values = mask * spherical_bessel_basis(l, roots[l, q-1], r) * ( spherical_harmonics(m, l, theta, phi) )  / np.sqrt(2 * l + 1) / ((2 * self.kernel_size + 1)**3 ) 
            kernel_values = torch.tensor(kernel_values, dtype=torch.float32).view(2 * self.kernel_size + 1,  2 * self.kernel_size + 1,  2 * self.kernel_size + 1)

            kernel[v_order, ...] = kernel_values

        """
        @flag: updated to this line.

        @todo: there is no real/imag anymore.
        """  

        # Construct tensor W. 

        if self.standard_layer:
            # 1st layer does not need W.
            W = torch.zeros(  (self.g_index.shape[0], self.g_index.shape[0], self.output_so3_grid.shape[0]), dtype=torch.float32) 

            for so3_index in range(self.output_so3_grid.shape[0]):

                beta, alpha, gamma = self.output_so3_grid[so3_index, :]

                for g_order_j, g_idx_j in enumerate(self.g_index):
                    b, j, n = g_idx_j[0], g_idx_j[1], g_idx_j[2]
                    for g_order_t, g_idx_t in enumerate(self.g_index):
                        b_, t, n_ = g_idx_t[0], g_idx_t[1], g_idx_t[2]
                        # 9 cases.
                        if j > 0 and n > 0: # case 1
                            if b == b_ and n == n_ :
                                W[g_order_j, g_order_t, so3_index] = wignerD(b, t, j, beta, alpha, gamma)
                            elif b == b_ and n == -n_:
                                W[g_order_j, g_order_t, so3_index] = (-1)** ( np.abs(j+n) ) * wignerD(b, t, -j, beta, alpha, gamma)
                        elif j > 0 and n < 0: # case 2
                            if b == b_ and n == n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = wignerD(b, t, j, beta, alpha, gamma)
                            elif b == b_ and n == -n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = (-1) ** ( np.abs(j + n) ) * wignerD(b, t, -j , beta, alpha, gamma)
                            elif b == b_ and n == n_ and t == 0:
                                W[g_order_j, g_order_t, so3_index] = (-1) ** (np.abs(j+1) ) * wignerD(b, 0, -j, beta, alpha, gamma)
                            elif b == b_ and b == -n_ and t == 0:
                                W[g_order_j, g_order_t, so3_index] = (-1) ** (np.abs(n)) * wignerD(b, 0, j, beta, alpha, gamma)
                        elif j > 0 and n == 0: # case 3
                            if b == b_ and n == n_:
                                W[g_order_j, g_order_t, so3_index] = wignerD(b, t, j, beta, alpha, gamma)
                            elif b == b_ and n == - n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = (-1) ** ( np.abs(j + n) ) * wignerD(b, t, -j , beta, alpha, gamma)
                        elif j < 0 and n > 0: # case 4
                            if b == b_ and n == n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] =  (-1) ** (np.abs(t - j)) * wignerD(b, -t, -j, beta, alpha, gamma)
                            elif b == b_ and n == -n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = (-1) ** (np.abs(t - n + 1)) * wignerD(b, -t, j, beta, alpha, gamma)
                            elif b == b_ and n == n_ and t == 0:
                                W[g_order_j, g_order_t, so3_index] = - wignerD(b, t, j, beta, alpha, gamma)
                            elif b == b_ and n == -n_ and t== 0:
                                W[g_order_j, g_order_t, so3_index] = (-1) ** (np.abs(n + j + 1)) * wignerD(b, t, -j, beta, alpha, gamma)
                        elif j < 0 and n < 0: # case 5
                            if b == b_ and n == n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = (-1) ** (np.abs(t - j)) * wignerD(b, -t, - j, beta, alpha, gamma)
                            elif b == b_ and n == - n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = (-1) ** (np.abs(t - n + 1)) * wignerD(b, - t, j, beta, alpha, gamma)
                            elif b == b_ and n == n_ and t == 0:
                                W[g_order_j, g_order_t, so3_index] = (-1)**(np.abs(j)) * wignerD(b, t, -j, beta, alpha, gamma )
                            elif b == b_ and n == -n_ and t == 0:
                                W[g_order_j, g_order_t, so3_index] = -(-1)**(np.abs(n)) * wignerD(b, t, j, beta, alpha, gamma )
                        elif j < 0 and n == 0: # case 6
                            if b == b_ and n == n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = (-1) ** (np.abs(t - j)) * wignerD(b, -t, -j, beta, alpha, gamma)
                            elif b == b_ and n == n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = (-1) ** (np.abs(t - n + 1)) * wignerD(b, - t, j,  beta, alpha, gamma)
                            elif b == b_ and n == n_ and t == 0:
                                W[g_order_j, g_order_t, so3_index] = - wignerD(b, t, j, beta, alpha, gamma )
                        elif j == 0 and n > 0: # case 7
                            if b == b_ and n == n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = wignerD(b, t, j, beta, alpha, gamma )
                            elif b == b_ and n == -n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = (-1)**(np.abs(j+n)) * wignerD(b, t, -j,beta, alpha, gamma  )
                            elif b == b_ and n == n_ and t  == 0:
                                W[g_order_j, g_order_t, so3_index] =  wignerD(b, t, j, beta, alpha, gamma )
                        elif j == 0 and n < 0: # case 8
                            if b == b_ and n == n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = (-1) ** (np.abs(t - j)) * wignerD(b, -t, -j, beta, alpha, gamma)
                            elif b == b_ and n == -n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = (-1) ** (np.abs(t - n + 1)) * wignerD(b, -t, j, beta, alpha, gamma)
                            elif b == b_ and n == n_ and t == 0:
                                W[g_order_j, g_order_t, so3_index] = wignerD(b, t, j, beta, alpha, gamma )
                        elif j == 0 and n == 0: # case 9
                            if b == b_ and n == n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = wignerD(b, t, j, beta, alpha, gamma )
                            elif b == b_ and n == -n_ and t is not 0:
                                W[g_order_j, g_order_t, so3_index] = (-1)**(np.abs(j+n)) * wignerD(b, t, -j,beta, alpha, gamma  )
                            elif b == b_ and n == n_ and t  == 0:
                                W[g_order_j, g_order_t, so3_index] =  wignerD(b, t, j, beta, alpha, gamma )
                        else:
                            pass
        # Construct tensor T.

        T = torch.zeros(  (self.v_index.shape[0], self.v_index.shape[0], self.output_so3_grid.shape[0]), dtype=torch.float32)  

        for so3_index in range(self.output_so3_grid.shape[0]):

            beta, alpha, gamma = self.output_so3_grid[so3_index, :]

            for v_order_m, v_idx_m in enumerate(self.v_index):
                q, l, m = v_idx_m[0], v_idx_m[1], v_idx_m[2]
                for v_order_s, v_idx_s in enumerate(self.v_index):
                    q_, l_, s = v_idx_s[0], v_idx_s[1], v_idx_s[2]
                    if q == q_ and l == l_:
                        # conj ( D_l^{s,m} (h) )
                        if m > 0 and s > 0:
                            T[v_order_m, v_order_s, so3_index] = (-1)**s * wignerD(l, s, -m, beta, alpha, gamma) + (-1)** ( np.abs(s -m) ) * wignerD(l, s , m,  beta, alpha, gamma) 
                        elif m > 0 and s < 0:
                            T[v_order_m, v_order_s, so3_index] = wignerD(l, s , -m,  beta, alpha, gamma) + (-1)**m * wignerD(l, s , m,   beta, alpha, gamma)
                        elif m > 0 and s == 0:
                            T[v_order_m, v_order_s, so3_index] =  np.sqrt(2) * (-1)**(m) * wignerD(l, 0, m, beta, alpha, gamma)
                        elif m < 0 and s > 0:
                            T[v_order_m, v_order_s, so3_index] = (-1)**(-m) * wignerD(l, -s, -m, beta, alpha, gamma) - wignerD(l, -s, m, beta, alpha, gamma)
                        elif m < 0 and s < 0:
                            T[v_order_m, v_order_s, so3_index] = -(-1)**(-s) * wignerD(l, -s, m, beta, alpha, gamma) + (-1)** ( np.abs(m_-m) ) * wignerD(l, -s, -m,  beta, alpha, gamma)
                        elif m < 0 and s == 0:
                            T[v_order_m, v_order_s, so3_index] = -np.sqrt(2) * wignerD(l, 0, m, beta, alpha, gamma)
                        elif m == 0 and s > 0:
                            T[v_order_m, v_order_s, so3_index] = ( (-1)**s * wignerD(l, s , -m, beta, alpha, gamma) + (-1)**( np.abs(s -m) ) * wignerD(l, s , m,  beta, alpha, gamma) ) / np.sqrt(2)
                        elif m == 0 and s < 0:
                            T[v_order_m, v_order_s, so3_index] = ( wignerD(l, s , -m,  beta, alpha, gamma) + (-1)**m * wignerD(l, s , m,   beta, alpha, gamma) ) / np.sqrt(2) 
                        elif m == 0 and s == 0:
                            T[v_order_m, v_order_s, so3_index] =  wignerD(l, 0, m, beta, alpha, gamma)
                        else:
                            pass 




        self.register_buffer('input_so3_weight',   input_so3_weight)
        self.register_buffer('output_so3_weight',  output_so3_weight)
        self.register_buffer('output_wigner_basis',output_wigner_basis)
        self.register_buffer('input_wigner_basis',input_wigner_basis)
        self.register_buffer('kernel',        kernel)
        
        if self.standard_layer:
            self.register_buffer('W', W)
            
        self.register_buffer('T', T)

        # set parameter for this layer. Requires grad.

        init_param   = torch.randn( self.in_channel, self.out_channel,  self.g_index.shape[0], self.v_index.shape[0])

        init_param   = init_param * torch.sqrt( 2 * torch.tensor(self.v_index[:, 1]).view(1,1,1, -1) + 1) # normalized harmonics

        init_param   = init_param * torch.sqrt( 2 * torch.tensor(self.g_index[:, 0]).view(1,1,-1, 1) + 1) # normalized wigners

        init_param  /= np.sqrt(in_channel)
        
        self.weights =  torch.nn.Parameter(init_param , requires_grad=True)
        
        
    def extra_repr(self):
        # Set the information of this module.
        return 'input_features={}, output_features={}, filter size={}, basis params=(q={}, l={}, b={}), so3_sampling={}, stride={}, padding={}, standard layer={}'.format(
            self.in_channel, self.out_channel, 2 * self.kernel_size + 1, self.sph_bessel_root, 
            self.sph_harm_index, self.wigner_index, self.so3_sampling, self.stride, 
            self.padding, self.standard_layer
        )        
               
        
    def compute_G(self, batch_images):
        # compute the G functions from batch_images of size:
        #   standard_layer:   {Batch Size (b)} x {Channel In (i)} x {Group Size (g)} x {H} x {W} x {D} x {real, imag}
        #   otheriwse:        {Batch Size (b)} x {Channel In (i)} x {H} x {W} x {D} x {real, imag}
        # 
        # Typical Input Batch Image Data 
        # standard layer: 1 x 8 x (26x8) x 32 x 32 x 32 x 2 (float32) = 0.7 GByte. (64x64x64 images ~ 5.6GBytes)

        
        # Step 1. 
        # Compute H functions through SO3 integral by einsum on {Group Size}. 
        #
        # Output is {Batch Size} x {Channel In} x {g_index} x {H} x {W} x {D} 
        # Typical Output Batch Data 1 x 8 x 84 x 32 x 32 x 32  (float32) = 0.16 GByte. (64x64x64 images ~ 1.2GBytes)
        # Should use GPU (by loading the data on GPU).

        H, W, D = batch_images.shape[-3], batch_images.shape[-2], batch_images.shape[-1] 

        if self.standard_layer:

            H_ =  torch.einsum('bighwd, kg->bikhwd', batch_images,  ( self.input_wigner_basis ) * self.input_so3_weight.view(1,-1))

        else:
            H_ =  batch_images


            
        # Step 2.
        # Convolution with Kernels (real/imag parts). The convolution takes 4 real convolutions since torch only supports real ops.

        # Batch Image Input From Step 1:
        #   standard_layer: [ {Batch Size} x {Channel In} x {g_index} ] x {H} x {W} x {D}
        #   otherwise:      [ {Batch Size} x {Channel In}  ]            x {H} x {W} x {D}

        # Conv3D filters: {Q x (L+1)^2 } x 1 x (filter_size) x (filter_size) x (filter_size). 
        # stride, padding are user-supplied.

        # Out: 
        #   standard_layer: {Batch Size} x {Channel In} x {g_index} x {v_index} x {H'} x {W'} x {D'} (depends on convolution arguments).
        #   otherwise:      {Batch Size} x {Channel In} x             {v_index} x {H'} x {W'} x {D'} (depends on convolution arguments).
        #
        # Typical Output Size: 1 x 8 x 84 x (3 x 35) x 32 x 32 x 32  float ~ 2.15 GBytes.

        output = torch.nn.functional.conv3d(input = H_.reshape(-1, 1, H, W, D ) , \
                                weight = self.kernel.view(-1, 1, 2 * self.kernel_size + 1 , 2 * self.kernel_size + 1 , 2 * self.kernel_size + 1 ), \
                                padding = self.padding, stride=self.stride)
        
        if self.standard_layer:
            return  output.view(batch_images.shape[0], \
                                batch_images.shape[1], \
                                self.g_index.shape[0], \
                                self.v_index.shape[0], \
                                output.shape[-3], \
                                output.shape[-2], \
                                output.shape[-1])
                    
        else:
            # non standard layer
            return  output.view(batch_images.shape[0], \
                                batch_images.shape[1], \
                                self.v_index.shape[0], \
                                output.shape[-3], \
                                output.shape[-2], \
                                output.shape[-1])     
        

    def non_standard_forward(self, batch_images):

        # both shape:
        #   standard:  {Batch Size (b)} x {Channel In (i)} x {g_index (K)} x {v_index (V)} x {H'} x {W'} x {D'} (depends on convolution arguments).
        #   otherwise: {Batch Size (b)} x {Channel In (i)}                 x {v_index (V)} x {H'} x {W'} x {D'} (depends on convolution arguments).
        
        with torch.no_grad(): # since this will be first layer and unrelates to parameter
            G = self.compute_G(batch_images) 

        # parameters:     {Channel In (i)} x {Channel Out (o)} x {g_index (K)} x {v_index (V)}

        # output shape:   {Batch Size (b)} x {Channel Out (o)} x {Group Size (g)} x {H'} x {W'} x {D'} x {real, imag}

        einsum_str1 = 'iokv, vVg->iokVg'
        einsum_str2 = 'iokVg, biVhwd->boghwd'

        output = torch.einsum(einsum_str1,    self.weights, self.T)
        output = torch.einsum(einsum_str2,    output, G)

        return output # Batch x Out Channel x Group x H x W x D


    def standard_forward(self, batch_images):
        
        # both shape:
        #   standard:  {Batch Size (b)} x {Channel In (i)} x {g_index (K)} x {v_index (V)} x {H'} x {W'} x {D'} (depends on convolution arguments).
        #   otherwise: {Batch Size (b)} x {Channel In (i)}                 x {v_index (V)} x {H'} x {W'} x {D'} (depends on convolution arguments).

        G = self.compute_G(batch_images) 
        
        # parameters:   {Channel In (i)} x {Channel Out (o)} x {g_index (K)} x {v_index (V)}

        # output shape: {Batch Size (b)} x {Channel Out (o)} x {Group Size (g)} x {H} x {W} x {D} x {real, imag}

        # There are 8 einsum to compute...

        # For memory efficiency, only 3 + (temp) terms needed.

        output = torch.einsum('iokv, kKg->ioKvg',          self.weights, self.W)
        output = torch.einsum('ioKvg, vVg->ioKVg',         output, self.T)
        output = torch.einsum('ioKVg, biKVhwd->boghwd',    output, G)

        return output # Batch x Out Channel x Group x H x W x D

    def forward(self, batch_images):
        if self.standard_layer:
            return self.standard_forward(batch_images)
        else:
            return self.non_standard_forward(batch_images)

    def check_frame_score(self):
        local_mesh_mat = np.zeros((self.v_index.shape[0], self.v_index.shape[0] ), dtype=complex )
        for row in range(self.v_index.shape[0]):
            for col in range(self.v_index.shape[0]):
                local_mesh_mat[row, col] = torch.sum(self.kernel[row, :, :, :] * self.kernel[col, :, :, :]).item()

        # normalize.
        for row in range(self.v_index.shape[0]):
            for col in range(self.v_index.shape[0]):
                local_mesh_mat[row, col] =  local_mesh_mat[row, col] / np.sqrt(local_mesh_mat[row, row].real   * local_mesh_mat[col, col].real)

        svd_vals = np.linalg.svd(local_mesh_mat, compute_uv=False)

        print( 'local 3d mesh score: (largest gap between 1 and singular values)', np.max( [np.abs(1-np.min(svd_vals)) , np.abs(1-np.max(svd_vals))] ) )

        local_wigner_mat = np.zeros((self.g_index.shape[0], self.g_index.shape[0]), dtype=complex)

        for row in range(self.g_index.shape[0]):
            for col in range(self.g_index.shape[0]):
                local_wigner_mat[row, col] = torch.sum( self.output_wigner_basis[row, :] * self.output_wigner_basis[col, :] * self.output_so3_weight ).item()  * (4*np.pi**3/ self.input_so3_grid.shape[0] )
        # normalize.
        for row in range(self.g_index.shape[0]):
            for col in range(self.g_index.shape[0]):
                local_wigner_mat[row, col] =  local_wigner_mat[row, col] / (np.sqrt(local_wigner_mat[row, row].real   * local_wigner_mat[col, col].real))

        wigner_svd_vals = np.linalg.svd(local_wigner_mat, compute_uv=False)
        print( 'group frame score: (largest gap between 1 and singular values)', np.max( [np.abs(1-np.min(wigner_svd_vals)) , np.abs(1-np.max(wigner_svd_vals))] ) )
