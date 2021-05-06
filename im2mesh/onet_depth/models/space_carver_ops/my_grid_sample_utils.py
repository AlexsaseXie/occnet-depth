import torch
from torch import nn

from . import _ext

class SpaceCarverGridSamplerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, interpolation_mode='DepthPlusFix', padding_mode='Zeros', align_corners=True, 
        invalid_value=0., fix_search_area=1):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Grid sampler special for space carver
        Parameters
        ----------
        input : torch.Tensor
            (B, C, H, W) tensor of input (usually depth map)
        grid : torch.Tensor
            (B, H_out, W_out, 2) tensor of grid index

        Returns
        -------
        output : torch.Tensor
            (B, C, H_out, W_out) output of sampled points
        """

        all_interpolation_mode = ['MaskPlusFix', 'DepthPlusFix', 'DepthBilinearPlusFix']
        assert interpolation_mode in all_interpolation_mode 
        interpolation_mode = all_interpolation_mode.index(interpolation_mode)

        all_padding_mode = ['Zeros', 'Border', 'Reflection']
        assert padding_mode in all_padding_mode
        padding_mode = all_padding_mode.index(padding_mode)

        output = _ext.my_grid_sampler_2d_forward_cuda(input, grid, interpolation_mode, padding_mode, align_corners, invalid_value, fix_search_area)

        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return ()

space_carver_grid_sampler = SpaceCarverGridSamplerFunction.apply

class SpaceCarverGridSamplerModule(nn.Module):
    def __init__(self, interpolation_mode='MaskPlusFix', padding_mode='Zeros', align_corners=True, 
        invalid_value=0., fix_search_area=1):
        all_interpolation_mode = ['MaskPlusFix', 'DepthPlusFix', 'DepthBilinearPlusFix']
        assert interpolation_mode in all_interpolation_mode 
        self.interpolation_mode = interpolation_mode

        all_padding_mode = ['Zeros', 'Border', 'Reflection']
        assert padding_mode in all_padding_mode
        self.padding_mode = padding_mode

        self.align_corners = align_corners
        self.invalid_value = invalid_value
        self.fix_search_area = fix_search_area

        self.function = SpaceCarverGridSamplerFunction()

    def forward(self, input, grid):
        return self.function(input, grid, 
            interpolation_mode=self.interpolation_mode, padding_mode=self.padding_mode,
            align_corners=self.align_corners, 
            invalid_value=self.invalid_value, fix_search_area=self.fix_search_area)
