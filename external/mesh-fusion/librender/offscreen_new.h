#ifndef LIBRENDER_OFFSCREEN_NEW_H
#define LIBRENDER_OFFSCREEN_NEW_H

#include "GL/glew.h"
#include "GL/gl.h"
#include "GL/glu.h"
#include "GL/glut.h"
#include <glm/glm.hpp>

class OffscreenGL_New {

public:
  OffscreenGL_New(int maxHeight, int maxWidth);
  ~OffscreenGL_New();

public:
  glm::mat4 view_mat;

private:
  static int glutWin;
  static bool glutInitialized;
  static bool shaderInitialized;
  static GLuint FBO;
  static GLuint rbo;
  GLuint normal_tex;
  GLuint vertex_tex;
  GLuint render_tex;

private:
  // shaders related
  GLuint frag_shader;
  GLuint vert_shader;
  static GLuint shader_program;
  // VAO
  GLuint VAO;
  GLuint VBO[4];
  // temp VBO
  float * t_color_array = nullptr;
  bool delete_color_flag = false;
  int * idx_array = nullptr;
  int face_count;
  // functions:
public:
  bool initialize_shaders();

  void prepare_vertex_info(
    float *vertex_array, float *color_array, float *normal_array, int fM, // 3 * N array
    bool use_color = true
  );

  void camera_setup(
    float zNear, float zFar, float *intrinsics, 
    unsigned int imgHeight, unsigned int imgWidth,
    float *camera_position, int index
  );

  void draw(
    unsigned char *imageBuffer, float *depthBuffer, bool *maskBuffer,
    float *normalBuffer, float *vertexBuffer,
    unsigned int imgHeight, unsigned int imgWidth,
    float *zNearFarV,
    int index, bool use_color = true
  );

  void copy_view_mat(float * viewMatBuffer, int index);
};


void render_mesh(
  // model information
  float *vertex_array, float *color_array, float *normal_array, int fM, // N * 3 array
  // use color or normal ?
  bool use_color, 
  // camera extrinsics
  float *camera_position, int T,  // T * 3 default : view (0,0,0), up(0,1,0)
  // camera intrinsics
  float *intrinsics, int *imgSizeV, float *zNearFarV,
  // output buffer
  unsigned char * imageBuffer, float *depthBuffer, bool *maskBuffer,
  float *normalBuffer, float *vertexBuffer, float * viewMatBuffer
);

void select_vertex(
  // input
  float * vertexBuffer, float * normalBuffer,
  int * imgSizeV,
  int fM, int T,
  // output
  float *point_cloud, int *point_cloud_size,
  float *face_normal_buffer, 
  int *stats
);

void select_faces(
  // input
  float * vertexBuffer, float * normalBuffer,
  int * imgSizeV,
  int fM, int T,
  // output
  bool *face_visible_buffer,
  float *face_normal_buffer, 
  int *stats
);

#endif
