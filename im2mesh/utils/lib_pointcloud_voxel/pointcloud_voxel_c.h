void find_nearby_grid_points(float * pointcloud, int pointcloud_N, int N,
    float * grid_points,
    long * calculate_grid_points_index, int * calculate_N,
    long * inside_grid_points_index, int * inside_grid_points_N,
    long * outside_grid_points_index, int * outside_grid_points_N, 
    int tolerance_K, float low, float high);

void find_nearby_grid_points_range(float * pointcloud, int pointcloud_N, int N,
    float * grid_points,
    long * calculate_grid_points_index, int * calculate_N,
    long * inside_grid_points_index, int * inside_grid_points_N,
    long * outside_grid_points_index, int * outside_grid_points_N, 
    float tolerance_K, float low, float high);


int to_grid_index(int N, float x, float low, float high);

float to_grid_index_float(int N, float x, float low, float high);