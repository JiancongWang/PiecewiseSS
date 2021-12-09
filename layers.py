import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear', dtype = torch.FloatTensor):
        super(SpatialTransformer, self).__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(dtype)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)


    def forward(self, src, flow, 
                return_phi = False, 
                add_grid = True,
                normalize = True):
        # new locations
        if add_grid:
            new_locs = self.grid + flow
            
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        if normalize:
            for i in range(len(shape)):
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if return_phi:
            if self.mode == "bilinear":
                return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode), new_locs
            else:
                return nnf.grid_sample(src, new_locs, mode=self.mode), new_locs
        else:
            if self.mode == "bilinear":
                return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
            else:
                return nnf.grid_sample(src, new_locs, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class MultiVecInt(nn.Module):
    ''' Integrate multiple velocity field via scvaling and squaring. 
    Then compose them together.
    '''
    
    def __init__(self, inshape, nsteps, num_V, 
				 scale_factor=1., dtype = torch.FloatTensor):
        super(MultiVecInt,self).__init__()
        
        self.ss = VecInt(inshape, nsteps)
        self.num_V = num_V
        self.inshape = inshape
        self.nsteps = nsteps
        self.transformer = SpatialTransformer(self.inshape, dtype = dtype)
        self.scale_factor = scale_factor
        
        
    def forward(self, vecs):
        # The input vecs should be of shape [bs, 3*num_V, x, y, z]/
        bs, c, x, y, z = vecs.shape
        assert c == 3*self.num_V, "Number of channel didn't match with number of Vs."
        
        phi = torch.zeros([bs, 3, x, y, z], 
                          device = vecs.device, 
                          dtype = vecs.dtype, 
                          requires_grad= True)
        
        for i in range(self.num_V):
            v_i = vecs[:, i*3:(i+1)*3, :, :, :] # chunk out the velocity field
            
            v_i = v_i * self.scale_factor
            
            # integrate the stationary velocity field
            # phi_i = self.ss(v_i) * self.scale_factor
            phi_i = self.ss(v_i) 
            
            # Compose the new phi_i onto the existing phi
            phi = phi + self.transformer(phi_i, phi)
        
        return phi



class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x
