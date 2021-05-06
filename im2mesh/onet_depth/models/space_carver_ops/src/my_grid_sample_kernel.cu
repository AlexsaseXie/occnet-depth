#ifndef _MY_GRID_SAMPLER_CUDA
#define _MY_GRID_SAMPLER_CUDA

#include <ATen/ATen.h>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

namespace space_carver {

namespace detail {
    enum class SpaceCarverInterpolation {MaskPlusFix, DepthPlusFix, DepthBilinearPlusFix };
}

using namespace at;
using namespace at::native;
using namespace at::cuda::detail;
//using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;
using detail::SpaceCarverInterpolation;

namespace {

    template<typename scalar_t, typename index_t>
    static __forceinline__ __device__
    bool is_invalid(index_t nearest_x, index_t nearest_y, index_t inp_H, index_t inp_W,
        scalar_t *inp_ptr_NC, index_t C, index_t inp_sC, index_t inp_sH, index_t inp_sW,
        scalar_t invalid_value) {
        // usually C == 1
        bool invalid_flag = true;
        if (within_bounds_2d(nearest_y, nearest_x, inp_H, inp_W)) {
            for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC) {
                if (inp_ptr_NC[nearest_y * inp_sH + nearest_x * inp_sW] != invalid_value) {
                    invalid_flag = false; break;
                }
            }
        }
        return invalid_flag;
    }

    // kernel function
    template <typename scalar_t, typename index_t>
    C10_LAUNCH_BOUNDS_1(1024)
    __global__ void my_grid_sampler_2d_kernel(
        const index_t nthreads,
        TensorInfo<scalar_t, index_t> input,
        TensorInfo<scalar_t, index_t> grid,
        TensorInfo<scalar_t, index_t> output,
        const scalar_t invalid_value, // change: add an invalid value
        const SpaceCarverInterpolation interpolation_mode,
        const GridSamplerPadding padding_mode,
        bool align_corners,
        const index_t fix_search_area // change: add neighbor search area
    ) {

        // beginning
        index_t C = input.sizes[1];
        index_t inp_H = input.sizes[2];
        index_t inp_W = input.sizes[3];
        index_t out_H = grid.sizes[1];
        index_t out_W = grid.sizes[2];
        index_t inp_sN = input.strides[0];
        index_t inp_sC = input.strides[1];
        index_t inp_sH = input.strides[2];
        index_t inp_sW = input.strides[3];
        index_t grid_sN = grid.strides[0];
        index_t grid_sH = grid.strides[1];
        index_t grid_sW = grid.strides[2];
        index_t grid_sCoor = grid.strides[3];
        index_t out_sN = output.strides[0];
        index_t out_sC = output.strides[1];
        index_t out_sH = output.strides[2];
        index_t out_sW = output.strides[3];

        CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
            const index_t w = index % out_W;
            const index_t h = (index / out_W) % out_H;
            const index_t n = index / (out_H * out_W);
            const index_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

            // get the corresponding input x, y co-ordinates from grid
            scalar_t x = grid.data[grid_offset];
            scalar_t y = grid.data[grid_offset + grid_sCoor];

            scalar_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
            scalar_t iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

            if (interpolation_mode == SpaceCarverInterpolation::DepthPlusFix
                || interpolation_mode == SpaceCarverInterpolation::DepthBilinearPlusFix 
                || interpolation_mode == SpaceCarverInterpolation::MaskPlusFix) {
                // judge if the corresponding pixel fall into background area of the depth map
                index_t nearest_x = static_cast<index_t>(::round(ix));
                index_t nearest_y = static_cast<index_t>(::round(iy));

                auto inp_ptr_NC = input.data + n * inp_sN;
                bool invalid_flag = is_invalid(
                    nearest_x, nearest_y, inp_H, inp_W, 
                    inp_ptr_NC, C, inp_sC, inp_sH, inp_sW, 
                    invalid_value
                );

                if (invalid_flag && fix_search_area > 0) {
                    // the corresponding pixel is in the background area
                    // try to search the neighbor area of 1
                    auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
                    
                    // initialize
                    for (index_t c = 0; c < C; c++) {
                        *(out_ptr_NCHW + out_sC * c) = static_cast<scalar_t>(0);
                    }

                    int count = 0;
                    // search neighbor pixels, find mean z value as the value at x.
                    for (index_t dx = -fix_search_area; dx <= fix_search_area; dx ++) {
                        for (index_t dy = -fix_search_area; dy <= fix_search_area; dy ++) {
                            if (dx == 0 && dy == 0) continue;
                            
                            // find mean z as the estimate depth at (x, y)
                            index_t x = nearest_x + dx;
                            index_t y = nearest_y + dy;
                            if (!is_invalid(x, y, inp_H, inp_W, 
                                inp_ptr_NC, C, inp_sC, inp_sH, inp_sW, 
                                invalid_value)) {
                                    count += 1;
                                    for (index_t c = 0; c < C; ++c) {
                                        *(out_ptr_NCHW + out_sC * c) += inp_ptr_NC[y * inp_sH + x * inp_sW + inp_sC * c];
                                    }
                            }
                        }
                    }

                    for (index_t c = 0; c < C; ++c) {
                        if (count > 0) 
                            *(out_ptr_NCHW + out_sC * c) /= static_cast<scalar_t>(count);
                        else 
                            *(out_ptr_NCHW + out_sC * c) = invalid_value;
                    }
                }
                else {
                    // already find the corresponding pixel with a valid z.
                    if (interpolation_mode == SpaceCarverInterpolation::DepthBilinearPlusFix) {
                        // try Bilinear Interpolation
                        index_t ix_nw = static_cast<index_t>(::floor(ix));
                        index_t iy_nw = static_cast<index_t>(::floor(iy));
                        index_t ix_ne = ix_nw + 1;
                        index_t iy_ne = iy_nw;
                        index_t ix_sw = ix_nw;
                        index_t iy_sw = iy_nw + 1;
                        index_t ix_se = ix_nw + 1;
                        index_t iy_se = iy_nw + 1;

                        bool all_valid = true;

                        // judge if valid
                        if (all_valid && is_invalid(ix_nw, iy_nw, inp_H, inp_W, 
                            inp_ptr_NC, C, inp_sC, inp_sH, inp_sW, 
                            invalid_value))
                            all_valid = false;
                        if (all_valid && is_invalid(ix_ne, iy_ne, inp_H, inp_W, 
                            inp_ptr_NC, C, inp_sC, inp_sH, inp_sW, 
                            invalid_value))
                            all_valid = false;
                        if (all_valid && is_invalid(ix_sw, iy_sw, inp_H, inp_W, 
                            inp_ptr_NC, C, inp_sC, inp_sH, inp_sW, 
                            invalid_value))
                            all_valid = false;
                        if (all_valid && is_invalid(ix_se, iy_se, inp_H, inp_W, 
                            inp_ptr_NC, C, inp_sC, inp_sH, inp_sW, 
                            invalid_value))
                            all_valid = false;

                        if (all_valid) {
                            // get surfaces to each neighbor:
                            scalar_t nw = (ix_se - ix)    * (iy_se - iy);
                            scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
                            scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
                            scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

                            // calculate bilinear weighted pixel value and set output pixel
                            // z interpolation: should be interpolated based on 1 / z
                            auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
                            for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
                                *out_ptr_NCHW = static_cast<scalar_t>(0);
                                
                                *out_ptr_NCHW += 1.0 / inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
                                *out_ptr_NCHW += 1.0 / inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
                                *out_ptr_NCHW += 1.0 / inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
                                *out_ptr_NCHW += 1.0 / inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
                                
                                *out_ptr_NCHW = 1.0 / *out_ptr_NCHW;
                            }
                        }
                        else {
                            // can only use Nearest interpolation
                            auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
                            for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
                                *out_ptr_NCHW = inp_ptr_NC[nearest_y * inp_sH + nearest_x * inp_sW];
                            }
                        }
                    }
                    else {
                        // Nearest interpolation
                        auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
                        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
                            *out_ptr_NCHW = inp_ptr_NC[nearest_y * inp_sH + nearest_x * inp_sW];
                        }
                    }
                }
            }

            /* reference code
            if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                // get NE, NW, SE, SW pixel values from (x, y)
                index_t ix_nw = static_cast<index_t>(::floor(ix));
                index_t iy_nw = static_cast<index_t>(::floor(iy));
                index_t ix_ne = ix_nw + 1;
                index_t iy_ne = iy_nw;
                index_t ix_sw = ix_nw;
                index_t iy_sw = iy_nw + 1;
                index_t ix_se = ix_nw + 1;
                index_t iy_se = iy_nw + 1;

                // get surfaces to each neighbor:
                scalar_t nw = (ix_se - ix)    * (iy_se - iy);
                scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
                scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
                scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

                // calculate bilinear weighted pixel value and set output pixel
                auto inp_ptr_NC = input.data + n * inp_sN;
                auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
                for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
                    *out_ptr_NCHW = static_cast<scalar_t>(0);
                    if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                        *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
                    }
                    if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                        *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
                    }
                    if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                        *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
                    }
                    if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                        *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
                    }
                }
            } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                index_t ix_nearest = static_cast<index_t>(::round(ix));
                index_t iy_nearest = static_cast<index_t>(::round(iy));

                // assign nearest neighor pixel value to output pixel
                auto inp_ptr_NC = input.data + n * inp_sN;
                auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
                for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
                    if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
                        *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
                    } else {
                        *out_ptr_NCHW = static_cast<scalar_t>(0);
                    }
                }
            } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {

                ix = grid_sampler_unnormalize(x, inp_W, align_corners);
                iy = grid_sampler_unnormalize(y, inp_H, align_corners);

                scalar_t ix_nw = ::floor(ix);
                scalar_t iy_nw = ::floor(iy);

                const scalar_t tx = ix - ix_nw;
                const scalar_t ty = iy - iy_nw;

                auto inp_ptr_NC = input.data + n * inp_sN;
                auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
                for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
                    scalar_t coefficients[4];

                    for (index_t i = 0; i < 4; ++i) {
                        coefficients[i] = cubic_interp1d(
                            get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                            get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                            get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                            get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                        tx);
                    }

                    *out_ptr_NCHW = cubic_interp1d(
                        coefficients[0],
                        coefficients[1],
                        coefficients[2],
                        coefficients[3],
                    ty);
                }
            }
            */
        }
    }
}


Tensor my_grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid,
    int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, float invalid_value, 
    int fix_search_area) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = grid.size(1);
    auto W = grid.size(2);
    auto output = at::empty({N, C, H, W}, input.options());
    int64_t count = N * H * W;
    if (count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "my_grid_sampler_2d_cuda", [&] {
            if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
                canUse32BitIndexMath(output)) {
                my_grid_sampler_2d_kernel<scalar_t>
                <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<int>(count),
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(grid),
                getTensorInfo<scalar_t, int>(output),
                static_cast<scalar_t>(invalid_value),
                static_cast<SpaceCarverInterpolation>(interpolation_mode),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners, 
                static_cast<int>(fix_search_area));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            } 
            else {
                my_grid_sampler_2d_kernel<scalar_t>
                <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                getTensorInfo<scalar_t, int64_t>(input),
                getTensorInfo<scalar_t, int64_t>(grid),
                getTensorInfo<scalar_t, int64_t>(output),
                static_cast<scalar_t>(invalid_value),
                static_cast<SpaceCarverInterpolation>(interpolation_mode),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners, 
                static_cast<int64_t>(fix_search_area));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        });
    }
    return output;
}

} // end of namespace space_carver

#endif
