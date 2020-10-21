#ifndef _EMD
#define _EMD

#include <vector>
#include <torch/extension.h>

//CUDA declarations
at::Tensor ApproxMatchForward(
    const at::Tensor xyz1,
    const at::Tensor xyz2);

at::Tensor MatchCostForward(
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const at::Tensor match);

std::vector<at::Tensor> MatchCostBackward(
    const at::Tensor grad_cost,
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const at::Tensor match);

#endif