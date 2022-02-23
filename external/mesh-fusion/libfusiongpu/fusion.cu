#include "gpu_common.h"

#include <cmath>
#include <vector>

__device__ __forceinline__ float atomicMinFloat(float * addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
       __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

  return old;
}

__device__ __forceinline__ float atomicMaxFloat(float * addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
       __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

  return old;
}

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


__global__ void kernel_inside_fusion(int n_pts, const Views views, Points points, int *counts) {
  CUDA_KERNEL_LOOP(idx, n_pts) {
    float x,y,z;
    int start = idx * points.c_dim_;
    x = points.data_[start + 0];
    y = points.data_[start + 1];
    z = points.data_[start + 2];

    int count_start = idx * 2;

    //int c_dim = points.c_dim_;
    //points.data_[start + (c_dim - 2)] = 0;
    //points.data_[start + (c_dim - 1)] = 0;

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
        if (dm_d <= 0 || vx_d <= dm_d) {
          counts[count_start] += 1;
          //points.data_[start + (c_dim - 2)] += 1;
        }
        else {
          counts[count_start + 1] += 1;
          //points.data_[start + (c_dim - 1)] += 1;
        }
      }
    }
  }
}

__global__ void kernel_tsdf_estimate(int n_pts, const Views views, Points points, float *sdf, float truncation) {
  CUDA_KERNEL_LOOP(idx, n_pts) {
    float x,y,z;
    int start = idx * points.c_dim_;
    x = points.data_[start + 0];
    y = points.data_[start + 1];
    z = points.data_[start + 2];

    int sdf_start = idx * 1;

    //int c_dim = points.c_dim_;
    //points.data_[start + (c_dim - 2)] = 0;
    //points.data_[start + (c_dim - 1)] = 0;

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

        // new sample
        float dist = dm_d - vx_d; 
        float truncated_dist = fminf(truncation, fmaxf(-truncation, dist));

        if(dm_d > 0 && dist >= -truncation) {
          // // valid views
          // if (dist <= truncation) {
          //   // true valid views
          //   if (valid_views == 0) {
          //     sdf[sdf_start] = truncated_dist;
          //     valid_views = 1;
          //   }
          //   else {
          //     valid_views ++;
          //     sdf[sdf_start] += truncated_dist;
          //   }
          // }
          // else {
          //   if (valid_views == 0) {
          //     sdf[sdf_start] = truncated_dist;
          //   }
          // }
          sdf[sdf_start] += truncated_dist;
          valid_views ++;
        }
      }
    }
    // after sample
    // if (valid_views != 0) {
    //   sdf[sdf_start] /= valid_views;
    // }
    // else if (valid_views == 0 && sdf[sdf_start] == 0) {
    //   sdf[sdf_start] = -truncation;
    // }

    if (valid_views > 0) {
      sdf[sdf_start] /= valid_views;
    }
    else {
      sdf[sdf_start] = -truncation;
    }
  }
}

__global__ void kernel_tsdf_inside_estimate(int n_pts, const Views views, Points points, int * counts, float *sdf, float truncation) {
  CUDA_KERNEL_LOOP(idx, n_pts) {
    float x,y,z;
    int start = idx * points.c_dim_;
    x = points.data_[start + 0];
    y = points.data_[start + 1];
    z = points.data_[start + 2];

    int sdf_start = idx * 1;
    int count_start = idx * 1;

    int valid_views = 0;
    counts[count_start] = 0;
    for(int vidx = 0; vidx < views.n_views_; ++vidx) {
      float ur, vr, vx_d;
      fusion_project(&views, vidx, x, y, z, ur,vr,vx_d);
      //NOTE: ur,vr,vx_d might differ to CPP (subtle differences in precision)

      int u = int(ur);
      int v = int(vr);

      if(u >= 0 && v >= 0 && u < views.cols_ && v < views.rows_) {
        int dm_idx = (vidx * views.rows_ + v) * views.cols_ + u;
        float dm_d = views.depthmaps_[dm_idx];

        // new sample
        float dist = dm_d - vx_d; 
        float truncated_dist = fminf(truncation, fmaxf(-truncation, dist));

        if (dm_d <= 0 || dist >= 0) {
          counts[count_start] += 1;
        }

        if(dm_d > 0 && dist >= -truncation) {
          sdf[sdf_start] += truncated_dist;
          valid_views ++;
        }
      }
    }
    if (valid_views > 0) {
      sdf[sdf_start] /= valid_views;
    }
    else {
      sdf[sdf_start] = -truncation;
    }
  }
}

__global__ void kernel_pointcloud_sdf_fusion(const Points pointcloud, Points query, 
  int* counts, float * sdf_records,
  int batch, int batch_count_int, float truncated_distance_square, int aggregate_type) {
  CUDA_KERNEL_LOOP(idx, batch_count_int) {
    long long start_number = (long long)batch * (long long)idx;      
    int query_idx = (int)(start_number / (long long)pointcloud.n_pts_);
    int pointcloud_idx = (int)(start_number % (long long)pointcloud.n_pts_);

    int query_start = query_idx * query.c_dim_;
    int count_start = query_idx * 2;
    int sdf_record_start = query_idx * 2;
    int pointcloud_start = pointcloud_idx * pointcloud.c_dim_;
    for (int vi=0;vi<batch;vi++) {
      if (query_idx >= query.n_pts_) break;

      float x,y,z;
      x = query.data_[query_start + 0];
      y = query.data_[query_start + 1];
      z = query.data_[query_start + 2];

      float pc_x, pc_y, pc_z;
      pc_x = pointcloud.data_[pointcloud_start + 0];
      pc_y = pointcloud.data_[pointcloud_start + 1];
      pc_z = pointcloud.data_[pointcloud_start + 2];

      float vec_x = x - pc_x;
      float vec_y = y - pc_y;
      float vec_z = z - pc_z;
      float sq_dis = vec_x * vec_x + vec_y * vec_y + vec_z * vec_z;

      if (sq_dis <= truncated_distance_square) {
        float normal_x = pointcloud.data_[pointcloud_start + 3];
        float normal_y = pointcloud.data_[pointcloud_start + 4];
        float normal_z = pointcloud.data_[pointcloud_start + 5];
        
        //float normal_len = sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);
        //normal_x /= normal_len;
        //normal_y /= normal_len;
        //normal_z /= normal_len;
        float sdf = vec_x * normal_x + vec_y * normal_y + vec_z * normal_z;
        if (aggregate_type == 0) {
          // separately consider positive sdf & negative sdf by min
          // this can be wrong when parallel
          if (sdf >= 0) {
            int count = atomicAdd(&(counts[count_start + 0]), 1);
            if (count == 0)
              atomicExch(&(sdf_records[sdf_record_start + 0]), sdf);
            else
              atomicMinFloat(&(sdf_records[sdf_record_start + 0]), sdf);
          }
          else {
            int count = atomicAdd(&(counts[count_start + 1]), 1);
            if (count == 0)
              atomicExch(&(sdf_records[sdf_record_start + 1]), sdf);
            else
              atomicMaxFloat(&(sdf_records[sdf_record_start + 1]), sdf);
          }
        }
        else if (aggregate_type == 1) {
          // separately consider positive sdf & negative sdf by average
          if (sdf >= 0) {
            atomicAdd(&(sdf_records[sdf_record_start + 0]), sdf);
            atomicAdd(&(counts[count_start + 0]), 1);
          }
          else {
            atomicAdd(&(sdf_records[sdf_record_start + 1]), sdf);
            atomicAdd(&(counts[count_start + 1]), 1);
          }
        }
      }

      pointcloud_idx ++;
      pointcloud_start += pointcloud.c_dim_;
      if (pointcloud_idx == pointcloud.n_pts_) {
        pointcloud_idx = 0;
        pointcloud_start = 0;
        query_idx ++;
        query_start += query.c_dim_;
        count_start += 2;
        sdf_record_start += 2;
      }
    }
  }
}

__global__ void kernel_pointcloud_sdf_fusion_var(const Points pointcloud, Points query, 
  int* counts, float * sdf_records,
  float truncated_distance_square, int aggregate_type) {

  int n = query.n_pts_;
  int m = pointcloud.n_pts_;

	const int batch = 512; // 512
	__shared__ float buf[batch*6];
  
  int sdf_record_n = 3;
  if (aggregate_type == 0) {
    sdf_record_n = 5;
  }
  else if (aggregate_type == 1) {
    sdf_record_n = 3;
  }

  for (int k2=0;k2<m;k2+=batch){
    int end_k=min(m,k2+batch)-k2;
    for (int j=threadIdx.x;j<end_k*6;j+=blockDim.x){
      buf[j]=pointcloud.data_[k2*6+j];
    }
    __syncthreads();
    CUDA_KERNEL_LOOP(j, n){
      float x = query.data_[j*3 + 0];
      float y = query.data_[j*3 + 1];
      float z = query.data_[j*3 + 2];
      for (int k=0;k<end_k;k++){
        float pc_x = buf[k*6 + 0];
        float pc_y = buf[k*6 + 1];
        float pc_z = buf[k*6 + 2];

        float vec_x = x - pc_x;
        float vec_y = y - pc_y;
        float vec_z = z - pc_z;
        float sq_dis = vec_x * vec_x + vec_y * vec_y + vec_z * vec_z;

        if (sdf_records[j*sdf_record_n + sdf_record_n - 1] == 0. || sq_dis < sdf_records[j*sdf_record_n + sdf_record_n - 1])
          sdf_records[j*sdf_record_n + sdf_record_n - 1] = sq_dis;

        if (sq_dis <= truncated_distance_square) {
          float normal_x = buf[k*6 + 3];
          float normal_y = buf[k*6 + 4];
          float normal_z = buf[k*6 + 5];
          
          float sdf = vec_x * normal_x + vec_y * normal_y + vec_z * normal_z;
          if (aggregate_type == 0) {
            // separately consider positive sdf & negative sdf by min
            if (sdf >= 0) {
              if (counts[j*2 + 0] == 0 || sdf_records[j*sdf_record_n + 0] > sq_dis) {
                sdf_records[j*sdf_record_n + 0] = sq_dis;
                sdf_records[j*sdf_record_n + 1] = sdf;
              }
              counts[j*2 + 0] += 1;
            }
            else {
              if (counts[j*2 + 1] == 0 || sdf_records[j*sdf_record_n + 2] > sq_dis) {
                sdf_records[j*sdf_record_n + 2] = sq_dis;
                sdf_records[j*sdf_record_n + 3] = sdf;
              }
              counts[j*2 + 1] += 1;
            }
          }
          else if (aggregate_type == 1) {
            // separately consider positive sdf & negative sdf by average
            if (sdf >= 0) {
              sdf_records[j*sdf_record_n + 0] += sdf;
              counts[j*2 + 0] += 1;
            }
            else {
              sdf_records[j*sdf_record_n + 1] += sdf;
              counts[j*2 + 1] += 1;
            }
          }
        }
      }
    }
    __syncthreads();
  }
}

__global__ void kernel_nearest_neighbor(const Points pointcloud, Points query, float * nn, int * id) {
  int n = query.n_pts_;
  int m = pointcloud.n_pts_;

	const int batch = 512; // 512
	__shared__ float buf[batch*6];

  for (int k2=0;k2<m;k2+=batch){
    int end_k=min(m,k2+batch)-k2;
    for (int j=threadIdx.x;j<end_k*6;j+=blockDim.x){
      buf[j]=pointcloud.data_[k2*6+j];
    }
    __syncthreads();
    CUDA_KERNEL_LOOP(j, n){
      float x = query.data_[j*query.c_dim_ + 0];
      float y = query.data_[j*query.c_dim_ + 1];
      float z = query.data_[j*query.c_dim_ + 2];
      for (int k=0;k<end_k;k++){
        float pc_x = buf[k*6 + 0];
        float pc_y = buf[k*6 + 1];
        float pc_z = buf[k*6 + 2];

        float vec_x = x - pc_x;
        float vec_y = y - pc_y;
        float vec_z = z - pc_z;
        float sq_dis = vec_x * vec_x + vec_y * vec_y + vec_z * vec_z;

        if (id[j] < 0 || nn[j] == 0. || sq_dis < nn[j]) {
          id[j] = k2 + k;
          nn[j] = sq_dis;
        }
      }
    }
    __syncthreads();
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

  int N = points.n_pts_ * 2;
  int * counts = device_malloc<int>(N);
  thrust::fill_n(thrust::device, counts, N, 0);

  kernel_inside_fusion <<< GET_BLOCKS(n_pts), CUDA_NUM_THREADS >>> (
    n_pts, views_gpu, points_gpu, counts
  );
  CUDA_POST_KERNEL_CHECK;

  int start = 0;
  int count_start = 0;
  int * counts_cpu = device_to_host_malloc<int>(counts, N);

  for (int i=0;i<points.n_pts_;i++) {
    points.data_[start + points.c_dim_ - 2] = counts_cpu[count_start];
    points.data_[start + points.c_dim_ - 1] = counts_cpu[count_start + 1];

    start += points.c_dim_;
    count_start += 2;
  }

  //points_to_cpu(points_gpu, points, false);
  views_free_gpu(views_gpu);
  points_free_gpu(points_gpu);
  device_free(counts);
  delete [] counts_cpu;
}

/* !!!BUG: need to copy buffer on cpu array */
void fusion_view_tsdf_estimation(const Views &views, Points &query, float truncated_distance) {
  Views views_gpu;
  views_to_gpu(views, views_gpu, true);
  Points query_gpu;
  points_to_gpu(query, query_gpu, true);

  int N = query.n_pts_ * 1;
  float * view_sdf = device_malloc<float>(N);
  thrust::fill_n(thrust::device, view_sdf, N, 0);

  kernel_tsdf_estimate <<< GET_BLOCKS(query.n_pts_), CUDA_NUM_THREADS >>> (
    query_gpu.n_pts_, views_gpu, query_gpu, view_sdf, truncated_distance
  );
  CUDA_POST_KERNEL_CHECK;

  int query_start = 0;
  for (int i=0;i<query.n_pts_;i++, query_start += query.c_dim_) {        
    query_gpu.data_[query_start + query.c_dim_ - 1] = view_sdf[i];
  }

  points_to_cpu(query_gpu, query, false);
  points_free_gpu(query_gpu);
  views_free_gpu(views_gpu);
  device_free(view_sdf);
}

void fusion_view_pc_tsdf_estimation(const Points& pointcloud, const Views& views, Points &query, float truncated_distance, int aggregate_type) {
  // truncated_distance: exp. 10 / 256
  // aggregate type: 0: min; 1: average
  
  Points pointcloud_gpu;
  points_to_gpu(pointcloud, pointcloud_gpu, true);
  Points query_gpu;
  points_to_gpu(query, query_gpu, true);
  // basic view fusion
  Views views_gpu;
  views_to_gpu(views, views_gpu, true);

  int N = query.n_pts_ * 1;
  float * view_sdf_gpu = device_malloc<float>(N);
  thrust::fill_n(thrust::device, view_sdf_gpu, N, 0.0f);

  kernel_tsdf_estimate <<< GET_BLOCKS(query.n_pts_), CUDA_NUM_THREADS >>> (
    query_gpu.n_pts_, views_gpu, query_gpu, view_sdf_gpu, truncated_distance
  );
  CUDA_POST_KERNEL_CHECK;
  
  // download
  float * view_sdf_cpu = device_to_host_malloc<float>(view_sdf_gpu, N);
  device_free<float>(view_sdf_gpu);

  // summarize
  int new_query_count = 0;
  for (int i=0;i<query.n_pts_;i++) {
    float v = view_sdf_cpu[i];
    //if (v != truncated_distance)
    //  printf("view_sdf:%f\n", v);
    if (v > -truncated_distance && v < truncated_distance) {
      new_query_count += 1;
    }
  }

  if (new_query_count == 0) return;

  Points new_query_cpu;
  new_query_cpu.n_pts_ = new_query_count;
  new_query_cpu.c_dim_ = 3;
  new_query_cpu.data_ = new float[new_query_count * 3];
  int index = 0;
  for (int i=0;i<N;i++) {
    if (view_sdf_cpu[i] > -truncated_distance && view_sdf_cpu[i] < truncated_distance) {
      for (int j=0;j<3;j++) {
        new_query_cpu.data_[index * 3 + j] = query.data_[i * query.c_dim_ + j];
      }
      index ++;
    } 
  }

  Points new_query_gpu;
  points_to_gpu(new_query_cpu, new_query_gpu, true);

  printf("new query gpu.n_pts_=%d\n", new_query_count);

  // point to point distance & sdf calculation
  int batch = 512;
  long long total_task = (long long)pointcloud.n_pts_ * (long long)new_query_gpu.n_pts_;
  long long batch_count = (total_task / (long long) batch) + 1;
  int batch_count_int = (int) batch_count;

  N = new_query_gpu.n_pts_ * 2;
  int * counts_gpu = device_malloc<int>(N);
  thrust::fill_n(thrust::device, counts_gpu, N, 0);

  N = new_query_gpu.n_pts_ * 2;
  float * sdf_records_gpu = device_malloc<float>(N);
  thrust::fill_n(thrust::device, sdf_records_gpu, N, 0.);

  // aggregate_type in (0, 1);
  kernel_pointcloud_sdf_fusion <<< GET_BLOCKS(batch_count_int), CUDA_NUM_THREADS >>> (
    pointcloud_gpu, new_query_gpu, counts_gpu, sdf_records_gpu, batch, 
    batch_count_int, 1e-4, aggregate_type
  );
  CUDA_POST_KERNEL_CHECK;


  //download
  int * counts_cpu = device_to_host_malloc<int>(counts_gpu, new_query_cpu.n_pts_ * 2);
  float * sdf_records_cpu = device_to_host_malloc<float>(sdf_records_gpu, new_query_cpu.n_pts_ * 2);
  device_free<int>(counts_gpu);
  device_free<float>(sdf_records_gpu);


  int start = 0;
  int count_start = 0;
  int sdf_record_start = 0;
  int c_dim = query.c_dim_;
  for (int i=0;i<query.n_pts_;i++) {
    if (view_sdf_cpu[i] > -truncated_distance && view_sdf_cpu[i] < truncated_distance) {
      if (counts_cpu[count_start] == 0 && counts_cpu[count_start + 1] == 0)  {
        query.data_[start + c_dim - 1] = view_sdf_cpu[i];
      }
      else {
        if (aggregate_type == 0) {
          // in most cases, eithor count_cpu[count_start] or count_cpu[count_start + 1] should be 0.
          if (counts_cpu[count_start] > counts_cpu[count_start + 1] * 5)
            query.data_[start + c_dim - 1] = sdf_records_cpu[sdf_record_start];
          else if (counts_cpu[count_start + 1] > counts_cpu[count_start] * 5)
            query.data_[start + c_dim - 1] = sdf_records_cpu[sdf_record_start + 1];
          else
            query.data_[start + c_dim - 1] = (sdf_records_cpu[sdf_record_start] + sdf_records_cpu[sdf_record_start + 1]) / 2.;
        }
        else {
          if (counts_cpu[count_start] > counts_cpu[count_start + 1] * 5)
            query.data_[start + c_dim - 1] = sdf_records_cpu[sdf_record_start] / (float)counts_cpu[count_start];
          else if (counts_cpu[count_start + 1] > counts_cpu[count_start] * 5)
            query.data_[start + c_dim - 1] = sdf_records_cpu[sdf_record_start + 1] / (float)counts_cpu[count_start + 1];
          else
            query.data_[start + c_dim - 1] = (sdf_records_cpu[sdf_record_start] + sdf_records_cpu[sdf_record_start + 1]) / 
              (float)(counts_cpu[count_start] + counts_cpu[count_start + 1]);
        }
      }
      count_start += 2;
      sdf_record_start += 2;
    }
    else {
      query.data_[start + c_dim - 1] = view_sdf_cpu[i];
    }
    start += query.c_dim_;
  }


  //points_to_cpu(query_gpu, query, false);


  // free space
  delete [] view_sdf_cpu;
  delete [] sdf_records_cpu;
  delete [] counts_cpu;
  points_free_gpu(pointcloud_gpu);
  points_free_gpu(query_gpu);
  points_free_gpu(new_query_gpu);
  delete [] new_query_cpu.data_;
  views_free_gpu(views_gpu);
}

void fusion_view_pc_tsdf_estimation_var(const Points& pointcloud, const Views& views, Points &query, float truncated_distance, int aggregate_type) {
  // truncated_distance: exp. 10 / 256
  // aggregate type: 0: min; 1: average
  
  Points pointcloud_gpu;
  points_to_gpu(pointcloud, pointcloud_gpu, true);
  Points query_gpu;
  points_to_gpu(query, query_gpu, true);
  // basic view fusion
  Views views_gpu;
  views_to_gpu(views, views_gpu, true);

  int N = query.n_pts_ * 1;
  float * view_sdf_gpu = device_malloc<float>(N);
  thrust::fill_n(thrust::device, view_sdf_gpu, N, 0.0f);
  int * view_count_gpu = device_malloc<int>(N);
  thrust::fill_n(thrust::device, view_count_gpu, N, 0);

  kernel_tsdf_inside_estimate <<< GET_BLOCKS(query.n_pts_), CUDA_NUM_THREADS >>> (
    query_gpu.n_pts_, views_gpu, query_gpu, view_count_gpu, view_sdf_gpu, truncated_distance
  );
  CUDA_POST_KERNEL_CHECK;
  
  // download
  float * view_sdf_cpu = device_to_host_malloc<float>(view_sdf_gpu, N);
  device_free<float>(view_sdf_gpu);
  int * view_count_cpu = device_to_host_malloc<int>(view_count_gpu, N);
  device_free<int>(view_count_gpu);

  // summarize
  int new_query_count = 0;
  for (int i=0;i<query.n_pts_;i++) {
    float v = view_sdf_cpu[i];
    //if (v != truncated_distance)
    //  printf("view_sdf:%f\n", v);
    if (v > -truncated_distance && v < truncated_distance) {
      new_query_count += 1;
    }
  }

  if (new_query_count == 0) return;

  Points new_query_cpu;
  new_query_cpu.n_pts_ = new_query_count;
  new_query_cpu.c_dim_ = 3;
  new_query_cpu.data_ = new float[new_query_count * 3];
  int index = 0;
  for (int i=0;i<N;i++) {
    if (view_sdf_cpu[i] > -truncated_distance && view_sdf_cpu[i] < truncated_distance) {
      for (int j=0;j<3;j++) {
        new_query_cpu.data_[index * 3 + j] = query.data_[i * query.c_dim_ + j];
      }
      index ++;
    } 
  }

  Points new_query_gpu;
  points_to_gpu(new_query_cpu, new_query_gpu, true);

  //printf("new query gpu.n_pts_=%d\n", new_query_count);

  // point to point distance & sdf calculation
  N = new_query_gpu.n_pts_ * 2;
  int * counts_gpu = device_malloc<int>(N);
  thrust::fill_n(thrust::device, counts_gpu, N, 0);

  int sdf_gpu_n = 3;
  float sq_thres = 4e-4; 
  if (aggregate_type == 0) {
    sdf_gpu_n = 5;
    //sq_thres = truncated_distance * truncated_distance ;
    sq_thres = 4e-4;
  }
  else if (aggregate_type == 1) { 
    sdf_gpu_n = 3;
    sq_thres = 4e-4;
  }

  N = new_query_gpu.n_pts_ * sdf_gpu_n;
  float * sdf_records_gpu = device_malloc<float>(N);
  thrust::fill_n(thrust::device, sdf_records_gpu, N, 0.);

  // aggregate_type in (0, 1);
  kernel_pointcloud_sdf_fusion_var <<< 32, 512 >>> (
    pointcloud_gpu, new_query_gpu, counts_gpu, sdf_records_gpu, 
    sq_thres,
    aggregate_type
  );
  CUDA_POST_KERNEL_CHECK;


  //download
  int * counts_cpu = device_to_host_malloc<int>(counts_gpu, new_query_cpu.n_pts_ * 2);
  float * sdf_records_cpu = device_to_host_malloc<float>(sdf_records_gpu, new_query_cpu.n_pts_ * sdf_gpu_n);
  device_free<int>(counts_gpu);
  device_free<float>(sdf_records_gpu);


  int start = 0;
  int view_count_start = 0;
  int count_start = 0;
  int sdf_record_start = 0;
  int c_dim = query.c_dim_;
  for (int i=0;i<query.n_pts_;i++) {
    if (view_sdf_cpu[i] > -truncated_distance && view_sdf_cpu[i] < truncated_distance) {
      bool side = true;
      if (counts_cpu[count_start] + counts_cpu[count_start + 1] <= 5)  {
        if (view_count_cpu[view_count_start] >= 10) {
          side = true;
        }
        else if (view_sdf_cpu[i] >= 0) {
          side = true;
        }
        else {
          side = false;
        }

        float d = sqrt(sdf_records_cpu[sdf_record_start + sdf_gpu_n - 1]);
        if (side == true) {
          query.data_[start + c_dim - 1] = d;
        }
        else {
          query.data_[start + c_dim - 1] = -d;
        }
      }
      else {
        if (aggregate_type == 0) {
          // in most cases, eithor count_cpu[count_start] or count_cpu[count_start + 1] should be 0.
          // if (counts_cpu[count_start] > counts_cpu[count_start + 1] * 5)
          //   query.data_[start + c_dim - 1] = sdf_records_cpu[sdf_record_start + 1];
          // else if (counts_cpu[count_start + 1] > counts_cpu[count_start] * 5)
          //   query.data_[start + c_dim - 1] = sdf_records_cpu[sdf_record_start + 3];
          // else
          //   query.data_[start + c_dim - 1] = (sdf_records_cpu[sdf_record_start + 1] + sdf_records_cpu[sdf_record_start + 3]) / 2.;
          if (view_count_cpu[view_count_start] >= 10) {
            side = true;
          }
          else if (counts_cpu[count_start] != 0 && counts_cpu[count_start + 1] != 0) {
            int large_count, small_count;
            float large_dis, small_dis;
            float large_dis_tsdf, small_dis_tsdf;
            if (sdf_records_cpu[sdf_record_start] < sdf_records_cpu[sdf_record_start + 2]) {
              large_dis = sqrt(sdf_records_cpu[sdf_record_start + 2]);
              large_dis_tsdf = sdf_records_cpu[sdf_record_start + 3];
              small_dis = sqrt(sdf_records_cpu[sdf_record_start + 0]);
              small_dis_tsdf = sdf_records_cpu[sdf_record_start + 1];
              large_count = counts_cpu[count_start + 1];
              small_count = counts_cpu[count_start];
              side = true;
            }
            else {
              large_dis = sqrt(sdf_records_cpu[sdf_record_start + 0]);
              large_dis_tsdf = sdf_records_cpu[sdf_record_start + 1];
              small_dis = sqrt(sdf_records_cpu[sdf_record_start + 2]);
              small_dis_tsdf = sdf_records_cpu[sdf_record_start + 3];
              large_count = counts_cpu[count_start];
              small_count = counts_cpu[count_start + 1];
              side = false;
            }

            if (small_dis * 1.05 >= large_dis) {
              if (large_count > small_count) {
                side = !side;
              }
            }
          }
          else if (counts_cpu[count_start] != 0) {
            side = true;
          }
          else if (counts_cpu[count_start + 1] != 0) {
            side = false;
          }

          if (side == true) {
            if (counts_cpu[count_start] == 0) {
              float d = sqrt(sdf_records_cpu[sdf_record_start + sdf_gpu_n - 1]);
              query.data_[start + c_dim - 1] = d;
            }
            else {
              // >= 0
              float n_tsdf = sdf_records_cpu[sdf_record_start + 1];
              float dis = sqrt(sdf_records_cpu[sdf_record_start]);
              if (dis * 0.866025 >= abs(n_tsdf)) 
                query.data_[start + c_dim - 1] = dis;
              else
                query.data_[start + c_dim - 1] = n_tsdf;
            }
          }
          else {
            if (counts_cpu[count_start + 1] == 0) {
              float d = sqrt(sdf_records_cpu[sdf_record_start + sdf_gpu_n - 1]);
              query.data_[start + c_dim - 1] = -d;
            }
            else {
              // <= 0
              float n_tsdf = sdf_records_cpu[sdf_record_start + 3];
              float dis = sqrt(sdf_records_cpu[sdf_record_start + 2]);
              if (dis * 0.866025 >= abs(n_tsdf)) 
                query.data_[start + c_dim - 1] = -dis;
              else
                query.data_[start + c_dim - 1] = n_tsdf;
            }
          }
        }
        else {
          if (counts_cpu[count_start] > counts_cpu[count_start + 1] * 5)
            query.data_[start + c_dim - 1] = sdf_records_cpu[sdf_record_start] / (float)counts_cpu[count_start];
          else if (counts_cpu[count_start + 1] > counts_cpu[count_start] * 5)
            query.data_[start + c_dim - 1] = sdf_records_cpu[sdf_record_start + 1] / (float)counts_cpu[count_start + 1];
          else
            query.data_[start + c_dim - 1] = (sdf_records_cpu[sdf_record_start] + sdf_records_cpu[sdf_record_start + 1]) / 
              (float)(counts_cpu[count_start] + counts_cpu[count_start + 1]);
        }
      }
      count_start += 2;
      sdf_record_start += sdf_gpu_n;
    }
    else {
      query.data_[start + c_dim - 1] = view_sdf_cpu[i];
    }
    start += query.c_dim_;
    view_count_start += 1;
  }


  //points_to_cpu(query_gpu, query, false);


  // free space
  delete [] view_sdf_cpu;
  delete [] view_count_cpu;
  delete [] sdf_records_cpu;
  delete [] counts_cpu;
  points_free_gpu(pointcloud_gpu);
  points_free_gpu(query_gpu);
  points_free_gpu(new_query_gpu);
  delete [] new_query_cpu.data_;
  views_free_gpu(views_gpu);
}

void fusion_nn_pc(const Points& pointcloud, Points &query) {
  Points pointcloud_gpu;
  points_to_gpu(pointcloud, pointcloud_gpu, true);
  Points query_gpu;
  points_to_gpu(query, query_gpu, true);

  int N = query_gpu.n_pts_;
  float * nn_gpu = device_malloc<float>(N);
  thrust::fill_n(thrust::device, nn_gpu, N, 0.);
  int * id_gpu = device_malloc<int>(N);
  thrust::fill_n(thrust::device, id_gpu, N, -1);

  kernel_nearest_neighbor <<< 32, 512 >>> (
    pointcloud_gpu, query_gpu, nn_gpu, id_gpu 
  );
  CUDA_POST_KERNEL_CHECK;

  float * nn_cpu = device_to_host_malloc<float>(nn_gpu, N);
  device_free<float>(nn_gpu);
  int * id_cpu = device_to_host_malloc<int>(id_gpu, N);
  device_free<int>(id_gpu);

  int c_dim = query.c_dim_; // == 5
  int start = 0;
  for (int i=0;i<query.n_pts_;i++) {
    query.data_[start + c_dim - 2] = nn_cpu[i];
    query.data_[start + c_dim - 1] = (float)id_cpu[i] + 0.1f;
    start += c_dim;
  }

  delete [] nn_cpu;
  delete [] id_cpu;
  points_free_gpu(query_gpu);
  points_free_gpu(pointcloud_gpu);
}
