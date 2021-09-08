#include "gpu_common.h"

#include <cmath>
#include <vector>



template <typename FusionFunctorT>
__global__ void kernel_fusion(int vx_res3, const Views views, const FusionFunctorT functor, float vx_size, Volume vol) {
  CUDA_KERNEL_LOOP(idx, vx_res3) {
    int d,h,w;
    fusion_idx2dhw(idx, vol.width_,vol.height_, d,h,w);
    float x,y,z;
    fusion_dhw2xyz(d,h,w, vx_size, x,y,z);

    functor.before_sample(&vol, d,h,w);
    bool run = true;
    int n_valid_views = 0;
    for(int vidx = 0; vidx < views.n_views_ && run; ++vidx) {
      float ur, vr, vx_d;
      fusion_project(&views, vidx, x,y,z, ur,vr,vx_d);
      //NOTE: ur,vr,vx_d might differ to CPP (subtle differences in precision)

      // Questionable procedure
      int u = int(ur + 0.5f);
      int v = int(vr + 0.5f);

      if(u >= 0 && v >= 0 && u < views.cols_ && v < views.rows_) {
        int dm_idx = (vidx * views.rows_ + v) * views.cols_ + u;
        float dm_d = views.depthmaps_[dm_idx];
        // if(d==103 && h==130 && w==153) printf("  dm_d=%f, dm_idx=%d, u=%d, v=%d, ur=%f, vr=%f\n", dm_d, dm_idx, u,v, ur,vr);
        run = functor.new_sample(&vol, vx_d, dm_d, d,h,w, &n_valid_views);
      }
    } // for vidx
    functor.after_sample(&vol, d,h,w, n_valid_views);
  }
}


__global__ void kernel_inside_fusion(int n_pts, const Views views, Points points) {
  CUDA_KERNEL_LOOP(idx, n_pts) {
    float x,y,z;
    int start = idx * points.c_dim_;
    x = points.data_[start + 0];
    y = points.data_[start + 1];
    z = points.data_[start + 2];

    int c_dim = points.c_dim_;
    points.data_[start + (c_dim - 2)] = 0;
    points.data_[start + (c_dim - 1)] = 0;

    int valid_views = 0;
    for(int vidx = 0; vidx < views.n_views_; ++vidx) {
      float ur, vr, vx_d;
      fusion_project(&views, vidx, x, y, z, ur,vr,vx_d);
      //NOTE: ur,vr,vx_d might differ to CPP (subtle differences in precision)

      int u = int(ur);
      int v = int(vr);

      if(u >= 0 && v >= 0 && u < views.cols_ && v < views.rows_) {
        int dm_idx = (vidx * views.rows_ + v) * views.cols_ + u;
        float dm_d = views.depthmaps_[dm_idx];
        // if(d==103 && h==130 && w==153) printf("  dm_d=%f, dm_idx=%d, u=%d, v=%d, ur=%f, vr=%f\n", dm_d, dm_idx, u,v, ur,vr);
        if (vx_d <= dm_d) {
          points.data_[start + (c_dim - 2)] += 1;
        }
        else {
          points.data_[start + (c_dim - 1)] += 1;
        }
      }
    }
  }
}


template <typename FusionFunctorT>
void fusion_gpu(const Views& views, const FusionFunctorT functor, float vx_size, Volume& vol) {
  Views views_gpu;
  views_to_gpu(views, views_gpu, true);
  Volume vol_gpu;
  volume_alloc_like_gpu(vol, vol_gpu);

  int vx_res3 = vol.depth_ * vol.height_ * vol.width_;
  kernel_fusion<<<GET_BLOCKS(vx_res3), CUDA_NUM_THREADS>>>(
    vx_res3, views_gpu, functor, vx_size, vol_gpu
  );
  CUDA_POST_KERNEL_CHECK;

  volume_to_cpu(vol_gpu, vol, false);

  views_free_gpu(views_gpu);
  volume_free_gpu(vol_gpu);
}

void fusion_projectionmask_gpu(const Views& views, float vx_size, bool unknown_is_free, Volume& vol) {
  ProjectionMaskFusionFunctor functor(unknown_is_free);
  fusion_gpu(views, functor, vx_size, vol);
}

void fusion_occupancy_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, Volume& vol) {
  OccupancyFusionFunctor functor(truncation, unknown_is_free);
  fusion_gpu(views, functor, vx_size, vol);
}

void fusion_tsdfmask_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, Volume& vol) {
  TsdfMaskFusionFunctor functor(truncation, unknown_is_free);
  fusion_gpu(views, functor, vx_size, vol);
}

void fusion_tsdf_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, Volume& vol) {
  TsdfFusionFunctor functor(truncation, unknown_is_free);
  fusion_gpu(views, functor, vx_size, vol);
}

void fusion_tsdf_strict_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, Volume& vol) {
  TsdfStrictFusionFunctor functor(truncation, unknown_is_free);
  fusion_gpu(views, functor, vx_size, vol);
}

void fusion_tsdf_range_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, Volume& vol) {
  TsdfRangeFusionFunctor functor(truncation, unknown_is_free);
  fusion_gpu(views, functor, vx_size, vol);
}


void fusion_tsdf_hist_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, float* bin_centers, int n_bins, bool unobserved_is_occupied, Volume& vol) {
  float* bin_centers_gpu = host_to_device_malloc(bin_centers, n_bins);
  TsdfHistFusionFunctor functor(truncation, unknown_is_free, bin_centers_gpu, n_bins, unobserved_is_occupied);
  fusion_gpu(views, functor, vx_size, vol);
  device_free(bin_centers_gpu);
}

void fusion_inside_gpu(const Views &views, int n_pts, Points & points) {
  //TODO
  Views views_gpu;
  views_to_gpu(views, views_gpu, true);
  Points points_gpu;
  points_to_gpu(points, points_gpu, true);

  kernel_inside_fusion <<< GET_BLOCKS(n_pts), CUDA_NUM_THREADS >>> (
    n_pts, views_gpu, points_gpu
  );
  CUDA_POST_KERNEL_CHECK;

  points_to_cpu(points_gpu, points, false);

  views_free_gpu(views_gpu);
  points_free_gpu(points_gpu);
}
