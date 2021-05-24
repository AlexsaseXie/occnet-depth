#ifndef LIBRENDER_OFFSCREEN_H
#define LIBRENDER_OFFSCREEN_H

#include "GL/glew.h"
#include "GL/gl.h"
#include "GL/glu.h"
#include "GL/glut.h"

class OffscreenGL {

public:
  OffscreenGL(int maxHeight, int maxWidth);
  ~OffscreenGL();

private:
  static int glutWin;
  static bool glutInitialized;
  GLuint FBO;
  GLuint depth_tex;
  GLuint normal_tex;
  GLuint vertex_tex;
  GLuint render_tex;

private:
  // shaders related
  GLuint frag_shader;
  GLuint vert_shader;
  GLuint shader_program;
  bool initialize_shaders();

  // VAO
  GLuint VAO;
  // temp VBO
  float * t_color_array = nullptr;
  bool delete_color_frag = false;
  int * idx_array = nullptr;
  void prepare_vertex_info(
    float *vertex_array, float *color_array, float *normal_array, int fM, // 3 * N array
    bool use_color = true
  );

private:
  void camera_setup(
    float zNear, float zFar, float *intrinsics, 
    unsigned int imgHeight, unsigned int imgWidth,
    float *camera_position, int index
  );

  void draw(
    unsigned char *imageBuffer, float *depthBuffer, bool *maskBuffer,
    float *normalBuffer, float *vertexBuffer,
    unsigned int imgHeight, unsigned int imgWidth,
    double *zNearFarV,
    int index, bool use_color = true,
  );
};


void render(
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
  float *normalBuffer, float *vertexBuffer 
);

#endif