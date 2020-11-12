#include <torch/extension.h>

#include "emd.h"
#include "expansion_penalty_module.h"
#include "MDS.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("emd_forward", &emd_forward, "emd forward (CUDA)");
  m.def("emd_backward", &emd_backward, "emd backward (CUDA)");

  m.def("expansion_penalty_forward", &expansion_penalty_forward, "expansion_penalty forward (CUDA)");
  m.def("expansion_penalty_backward", &expansion_penalty_backward, "expansion_penalty backward (CUDA)");

  m.def("minimum_density_sampling", &minimum_density_sampling, "minimum_density_sampling (CUDA)");
  m.def("gather_points_grad", &gather_points_grad, "gather_points_grad (CUDA)");
  m.def("gather_points", &gather_points, "gather_points (CUDA)");
}