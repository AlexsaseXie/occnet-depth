#pragma once
#include <torch/extension.h>

at::Tensor minimum_density_sampling(at::Tensor points, 
                                const int nsamples, at::Tensor mean_mst_length, at::Tensor output);

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);

at::Tensor gather_points(at::Tensor points, at::Tensor idx);
