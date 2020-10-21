#include "chamfer_distance.h"
#include "emd.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cd_forward", &chamfer_distance_forward, "ChamferDistance forward");
    m.def("cd_forward_cuda", &chamfer_distance_forward_cuda, "ChamferDistance forward (CUDA)");
    m.def("cd_backward", &chamfer_distance_backward, "ChamferDistance backward");
    m.def("cd_backward_cuda", &chamfer_distance_backward_cuda, "ChamferDistance backward (CUDA)");
    m.def("approxmatch_forward", &ApproxMatchForward,"ApproxMatch forward (CUDA)");
    m.def("matchcost_forward", &MatchCostForward,"MatchCost forward (CUDA)");
    m.def("matchcost_backward", &MatchCostBackward,"MatchCost backward (CUDA)");
}