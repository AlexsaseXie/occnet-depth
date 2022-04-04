#include<vector>
#include<iostream>
#include<cmath>
#include<queue>
using namespace std;
#include "pointcloud_voxel_c.h"

class Point {
public:
    int x;
    int y;
    int z;
public:
    Point(int ax=0,int ay=0, int az=0): x(ax), y(ay), z(az) {}
    ~Point() {}
};

void find_nearby_grid_points(float * pointcloud, int pointcloud_N, int N,
    float * grid_points,
    long * calculate_grid_points_index, int * calculate_N,
    long * inside_grid_points_index, int * inside_grid_points_N,
    long * outside_grid_points_index, int * outside_grid_points_N, 
    int tolerance_K, float low, float high) {

    // cout << "save init" << endl;

    vector<vector<vector<int>>> visited(N);
    
    // low : -0.5, high: 0.5
    for (int i=0;i<N;i++) {
        visited[i] = vector<vector<int>>(N);
        for (int j=0;j<N;j++){
            visited[i][j] = vector<int>(N, 0);
        }
    }

    // cout << "save reserve space" << endl;
    // cout << "toler_K:" << tolerance_K <<endl;
    //float high = 0.5;
    //float low = -0.5;
    float grid_len = (high - low) / N;

    
    // set k
    float * pc_pt = pointcloud;
    for (int i=0;i<pointcloud_N;i++, pc_pt+=3) {
        float pc_x = *pc_pt;
        float pc_y = *(pc_pt+1);
        float pc_z = *(pc_pt+2);
        
        int x_i = to_grid_index(N, pc_x, low, high);
        int y_i = to_grid_index(N, pc_y, low, high);
        int z_i = to_grid_index(N, pc_z, low, high);

        // if (i <= 5) {
        //     cout << "point cloud:" << pc_x << "," << pc_y << "," << pc_z << endl;
        //     cout << "->id" << x_i << "," << y_i << "," << z_i << endl;
        // }

        for (int dx=-tolerance_K;dx<=tolerance_K;dx++) {
            for (int dy=-tolerance_K;dy<=tolerance_K;dy++) {
                for (int dz=-tolerance_K;dz<=tolerance_K;dz++) {
                    int nx = x_i + dx;
                    int ny = y_i + dy;
                    int nz = z_i + dz;
                    if (nx >= 0 && nx <= N-1 && ny >= 0 && ny <= N-1 && nz >= 0 && nz <= N-1) {
                        visited[nx][ny][nz] = 1;
                    }
                }
            }
        }
    }

    // cout << "save calc surface" << endl;

    int calculate_grid_points_count = 0;
    long *calculate_index_pt = calculate_grid_points_index;
    float x = low + grid_len / 2.0;
    float y = x;
    float z = x;
    float * grid_pt = grid_points;
    for (int i=0;i<N;i++, x+=grid_len) {
        y = low + grid_len / 2.0;
        for (int j=0;j<N;j++, y+=grid_len) {
            z = low + grid_len / 2.0;
            for (int k=0;k<N;k++, z+=grid_len) {
                *grid_pt = x;
                *(grid_pt + 1) = y;
                *(grid_pt + 2) = z;

                if (visited[i][j][k] == 1) {
                    calculate_grid_points_count ++;
                    *calculate_index_pt = i;
                    *(calculate_index_pt + 1) = j;
                    *(calculate_index_pt + 2) = k;
                    calculate_index_pt += 3;

                    *(grid_pt + 3) = 1;
                }
                grid_pt += 4;
            }
        }
    }
    *calculate_N = calculate_grid_points_count;

    // cout << "save record surface" << endl;

    queue<Point> q;
    queue<Point> nq;

    int outside_grid_points_count = 0;
    int inside_grid_points_count = 0;
    long * inside_pt = inside_grid_points_index;
    long * outside_pt = outside_grid_points_index;
    int ds[6][3] = {{-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}};
    for (int i=0;i<N;i++) {
        for (int j=0;j<N;j++) {
            for (int k=0;k<N;k++) {
                if (visited[i][j][k] == 0) {
                    q.push(Point(i,j,k));
                    visited[i][j][k] = 2;
                    bool from_edge = false;
                    while(!q.empty()) {
                        Point cur = q.front();
                        q.pop();
                        nq.push(cur);

                        if (cur.x == 0 || cur.x == N - 1 || 
                            cur.y == 0 || cur.y == N - 1 ||
                            cur.z == 0 || cur.z == N - 1) {
                                from_edge = true;
                        }
                        
                        for (int t=0;t<6;t++) {
                            int nx = cur.x + ds[t][0];
                            int ny = cur.y + ds[t][1];
                            int nz = cur.z + ds[t][2];
                            if (nx >= 0 && nx <= N-1 && ny >= 0 && ny <= N-1 && nz >= 0 && nz <= N-1) {
                                if (visited[nx][ny][nz] == 0) {
                                    visited[nx][ny][nz] = 2;
                                    q.push(Point(nx, ny, nz));
                                }
                            }
                        }
                    }

                    if (from_edge) {
                        while(!nq.empty()) {
                            Point cur = nq.front();
                            nq.pop();

                            *outside_pt = cur.x;
                            *(outside_pt + 1) = cur.y;
                            *(outside_pt + 2) = cur.z;
                            outside_pt += 3;

                            outside_grid_points_count ++;

                            grid_points[(cur.x * N * N + cur.y * N + cur.z) * 4 + 3] = 0; 
                        }
                    }
                    else {
                        while(!nq.empty()) {
                            Point cur = nq.front();
                            nq.pop();

                            *inside_pt = cur.x;
                            *(inside_pt + 1) = cur.y;
                            *(inside_pt + 2) = cur.z;
                            inside_pt += 3;

                            inside_grid_points_count ++;
                            grid_points[(cur.x * N * N + cur.y * N + cur.z) * 4 + 3] = 2; 
                        }
                    }
                }
                else continue;
            }
        }
    }
    *outside_grid_points_N = outside_grid_points_count;
    *inside_grid_points_N = inside_grid_points_count;
}

void find_nearby_grid_points_range(float * pointcloud, int pointcloud_N, int N,
    float * grid_points,
    long * calculate_grid_points_index, int * calculate_N,
    long * inside_grid_points_index, int * inside_grid_points_N,
    long * outside_grid_points_index, int * outside_grid_points_N, 
    float tolerance_K, float low, float high) {

    // cout << "save init" << endl;

    vector<vector<vector<int>>> visited(N);
    
    // low : -0.5, high: 0.5
    for (int i=0;i<N;i++) {
        visited[i] = vector<vector<int>>(N);
        for (int j=0;j<N;j++){
            visited[i][j] = vector<int>(N, 0);
        }
    }

    // cout << "save reserve space" << endl;
    // cout << "toler_K:" << tolerance_K <<endl;
    //float high = 0.5;
    //float low = -0.5;
    float grid_len = (high - low) / N;

    
    // set k
    float * pc_pt = pointcloud;
    for (int i=0;i<pointcloud_N;i++, pc_pt+=3) {
        float pc_x = *pc_pt;
        float pc_y = *(pc_pt+1);
        float pc_z = *(pc_pt+2);
        
        int x_low = round(to_grid_index_float(N, pc_x - tolerance_K, low, high));
        int x_high = round(to_grid_index_float(N, pc_x + tolerance_K, low, high));
        int y_low = round(to_grid_index_float(N, pc_y - tolerance_K, low, high));
        int y_high = round(to_grid_index_float(N, pc_y + tolerance_K, low, high));
        int z_low = round(to_grid_index_float(N, pc_z - tolerance_K, low, high));
        int z_high = round(to_grid_index_float(N, pc_z + tolerance_K, low, high));

        // if (i <= 5) {
        //     cout << "point cloud:" << pc_x << "," << pc_y << "," << pc_z << endl;
        //     cout << "->id" << x_i << "," << y_i << "," << z_i << endl;
        // }

        for (int nx=x_low;nx<=x_high;nx++) {
            for (int ny=y_low;ny<=y_high;ny++) {
                for (int nz=z_low;nz<=z_high;nz++) {
                    if (nx >= 0 && nx <= N-1 && ny >= 0 && ny <= N-1 && nz >= 0 && nz <= N-1) {
                        visited[nx][ny][nz] = 1;
                    }
                }
            }
        }
    }

    // cout << "save calc surface" << endl;

    int calculate_grid_points_count = 0;
    long *calculate_index_pt = calculate_grid_points_index;
    float x = low + grid_len / 2.0;
    float y = x;
    float z = x;
    float * grid_pt = grid_points;
    for (int i=0;i<N;i++, x+=grid_len) {
        y = low + grid_len / 2.0;
        for (int j=0;j<N;j++, y+=grid_len) {
            z = low + grid_len / 2.0;
            for (int k=0;k<N;k++, z+=grid_len) {
                *grid_pt = x;
                *(grid_pt + 1) = y;
                *(grid_pt + 2) = z;

                if (visited[i][j][k] == 1) {
                    calculate_grid_points_count ++;
                    *calculate_index_pt = i;
                    *(calculate_index_pt + 1) = j;
                    *(calculate_index_pt + 2) = k;
                    calculate_index_pt += 3;

                    *(grid_pt + 3) = 1;
                }
                grid_pt += 4;
            }
        }
    }
    *calculate_N = calculate_grid_points_count;

    // cout << "save record surface" << endl;

    queue<Point> q;
    queue<Point> nq;

    int outside_grid_points_count = 0;
    int inside_grid_points_count = 0;
    long * inside_pt = inside_grid_points_index;
    long * outside_pt = outside_grid_points_index;
    int ds[6][3] = {{-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}};
    for (int i=0;i<N;i++) {
        for (int j=0;j<N;j++) {
            for (int k=0;k<N;k++) {
                if (visited[i][j][k] == 0) {
                    q.push(Point(i,j,k));
                    visited[i][j][k] = 2;
                    bool from_edge = false;
                    while(!q.empty()) {
                        Point cur = q.front();
                        q.pop();
                        nq.push(cur);

                        if (cur.x == 0 || cur.x == N - 1 || 
                            cur.y == 0 || cur.y == N - 1 ||
                            cur.z == 0 || cur.z == N - 1) {
                                from_edge = true;
                        }
                        
                        for (int t=0;t<6;t++) {
                            int nx = cur.x + ds[t][0];
                            int ny = cur.y + ds[t][1];
                            int nz = cur.z + ds[t][2];
                            if (nx >= 0 && nx <= N-1 && ny >= 0 && ny <= N-1 && nz >= 0 && nz <= N-1) {
                                if (visited[nx][ny][nz] == 0) {
                                    visited[nx][ny][nz] = 2;
                                    q.push(Point(nx, ny, nz));
                                }
                            }
                        }
                    }

                    if (from_edge) {
                        while(!nq.empty()) {
                            Point cur = nq.front();
                            nq.pop();

                            *outside_pt = cur.x;
                            *(outside_pt + 1) = cur.y;
                            *(outside_pt + 2) = cur.z;
                            outside_pt += 3;

                            outside_grid_points_count ++;

                            grid_points[(cur.x * N * N + cur.y * N + cur.z) * 4 + 3] = 0; 
                        }
                    }
                    else {
                        while(!nq.empty()) {
                            Point cur = nq.front();
                            nq.pop();

                            *inside_pt = cur.x;
                            *(inside_pt + 1) = cur.y;
                            *(inside_pt + 2) = cur.z;
                            inside_pt += 3;

                            inside_grid_points_count ++;
                            grid_points[(cur.x * N * N + cur.y * N + cur.z) * 4 + 3] = 2; 
                        }
                    }
                }
                else continue;
            }
        }
    }
    *outside_grid_points_N = outside_grid_points_count;
    *inside_grid_points_N = inside_grid_points_count;
}

int to_grid_index(int N, float x, float low, float high) {
    float grid_len = (high - low) / (float)N;
    int index = floor((x - low) / grid_len);

    return index;
}

float to_grid_index_float(int N, float x, float low, float high) {
    float grid_len = (high - low) / (float)N;
    float index = (x - low) / grid_len;

    return index;
}