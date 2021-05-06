#include <torch/extension.h>
#include <torch/torch.h>

namespace space_carver {

using namespace at;
Tensor my_grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid,
    int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, float invalid_value, 
    int fix_search_area);

}

using namespace space_carver;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_grid_sampler_2d_forward_cuda", my_grid_sampler_2d_cuda, "My grid sampler forward cuda");
}
