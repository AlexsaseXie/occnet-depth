#pragma once
#include <torch/extension.h>

int emd_forward(at::Tensor xyz1, at::Tensor xyz2, 
                    at::Tensor dist, at::Tensor assignment, at::Tensor price, 
	                at::Tensor assignment_inv, at::Tensor bid, 
                    at::Tensor bid_increments, at::Tensor max_increments,
	                at::Tensor unass_idx, at::Tensor unass_cnt, 
                    at::Tensor unass_cnt_sum, at::Tensor cnt_tmp, 
                    at::Tensor max_idx, float eps, int iters);

int emd_backward(at::Tensor xyz1, at::Tensor xyz2, 
                    at::Tensor gradxyz, at::Tensor graddist, at::Tensor idx);