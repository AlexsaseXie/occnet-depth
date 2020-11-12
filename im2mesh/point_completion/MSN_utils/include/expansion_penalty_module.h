#pragma once
#include <torch/extension.h>

int expansion_penalty_forward(at::Tensor xyz, 
                                int primitive_size, 
                                at::Tensor father, 
                                at::Tensor dist, 
                                double alpha, 
                                at::Tensor neighbor, 
                                at::Tensor cost, 
                                at::Tensor mean_mst_length);

int expansion_penalty_backward(at::Tensor xyz, 
                                at::Tensor gradxyz, 
                                at::Tensor graddist, 
                                at::Tensor idx);