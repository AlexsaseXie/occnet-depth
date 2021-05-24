#include "offscreen_new.h"
#include <cstdio>
#include <fstream>
#include "glm/glm/glm.hpp"

int OffscreenGL::glutWin = -1;
bool OffscreenGL::glutInitialized = false;

OffscreenGL::OffscreenGL(int maxHeight, int maxWidth) {

  if (!glutInitialized) {
    int argc = 1;
    char *argv = "test";
    glutInit(&argc, &argv);
    glutInitialized = true;
  }

  glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(maxWidth, maxHeight);

  // create or set window & off-screen framebuffer
  if (glutWin < 0) {

    glutWin = glutCreateWindow("OpenGL");
    glutHideWindow();
    glewInit();

    // FBO textures
    glGenTextures(1, &this->render_tex);
    glBindTexture(GL_TEXTURE_2D, this->render_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, maxWidth, maxHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glGenTextures(1, &this->depth_tex);
    glBindTexture(GL_TEXTURE_2D, this->depth_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, maxWidth, maxHeight, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, 0);

    glGenTextures(1, &this->normal_tex);
    glBindTexture(GL_TEXTURE_2D, this->normal_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, maxWidth, maxHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glGenTextures(1, &this->vertex_tex);
    glBindTexture(GL_TEXTURE_2D, this->vertex_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, maxWidth, maxHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    
    // FBO
    glGenFramebuffers(1, &this->FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, this->FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->vertex_tex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, this->normal_tex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, this->render_tex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER,  GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, this->depth_tex, 0);
    
    GLenum draw_buffers = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    
    glDrawBuffers(3, draw_buffers);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
  } else {
    // WARNING: shouldn't be called
    glutSetWindow(glutWin);
  }
}

OffscreenGL::~OffscreenGL() {
  // free space
  if (this->delete_color_frag)
    delete [] this->t_color_array;
  
  delete [] this->idx_array;
}

// load shaders
static const GLchar* getFileData( const char * path ) {
	FILE* infile;
	fopen_s(&infile, path, "rb");
	if(!infile)
	{
		return NULL;
	}
	fseek(infile, 0, SEEK_END);
	int len = ftell(infile);
	fseek(infile, 0, SEEK_SET);

	GLchar *source = new GLchar[len + 1];
	fread(source, 1, len, infile);
	fclose(infile);
	source[len] = 0;
	return const_cast<const GLchar *>(source);
}

bool compileShader( GLuint &shader_id, const char *path ) {
	const GLchar *shader_code = getFileData(path);
	if(strlen(shader_code) <= 0 )
		return 0;

	GLint Result = GL_FALSE;
	GLint InfoLogLength;

	glShaderSource(shader_id, 1, &shader_code, NULL);
	glCompileShader(shader_id);
	delete []shader_code;

	glGetShaderiv(shader_id, GL_COMPILE_STATUS, &Result);
	if ( !Result ){
		glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &InfoLogLength);
		std::vector<char> ShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(shader_id, InfoLogLength, NULL, &ShaderErrorMessage[0]);
		//printf("%s\n", &ShaderErrorMessage[0]);
		log("%s", &ShaderErrorMessage[0]);
		return false;
	}

	return true;
}

bool OffscreenGL::initialize_shaders() {
  this->vert_shader = glCreateShader(GL_VERTEX_SHADER);
  this->frag_shader = glCreateShader(GL_FRAGMENT_SHADER);

  bool vert_flag = compileShader(this->vert_shader, "./shaders/vert_shader.vs");
  bool frag_flag = compileShader(this->frag_shader, "./shaders/frag_shader.fs");

  if (!vert_flag || !frag_flag) 
    return false;

  this->shader_program = glCreateProgram();
  glAttachShader(this->shader_program, this->vert_shader);
  glAttachShader(this->shader_program, this->frag_shader);

  // bind location
  // vertex shader:
  glBindAttribLocation(this->shader_program, 0, "vertex");
  glBindAttribLocation(this->shader_program, 1, "normal");
  glBindAttribLocation(this->shader_program, 2, "color");
  glBindAttribLocation(this->shader_program, 3, "id");
  // fragment shader:
  glBindFragDataLocation(this->shader_program, 0, "frag_world_xyz");
  glBindFragDataLocation(this->shader_program, 1, "frag_normal");
  glBindFragDataLocation(this->shader_program, 2, "frag_color");

  glLinkProgram(this->shader_program);
  glUseProgram(this->shader_program);

  return true;
}

// vertex info : VAO & VBO
void OffscreenGL::prepare_vertex_info(
  float *vertex_array, float *color_array, float *normal_array, int fM, // 3 * N array
  bool use_color=true) {
  // color
  if (!use_color) {
    this->t_color_array = new float [fM * 3];
    for (int i=0; i < fM;i+=3) {
      this->t_color_array[i] = 1.0;
      this->t_color_array[i+1] = 0.;
      this->t_color_array[i+2] = 0.;
    }

    this->delete_color_flag = true;
  }
  else 
    this->t_color_array = color_array;
    
  // idx array
  this->idx_array = new int [fM * 3];
  for (int i=0;i<fM;i+=1) {
    this->idx_array[i * 3] = i;
    this->idx_array[i * 3 + 1] = i;
    this->idx_array[i * 3 + 2] = i;
  }

  // vertex, normal, color, idx VBOs
  GLuint VBO[4];
  glGenBuffers(4, VBO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * fM * 3, vertex_array, GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * fM * 3, normal_array, GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * fM * 3, this->t_color_array, GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, VBO[3]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(int) * fM * 3, this->idx_array, GL_STATIC_DRAW);

  // VAO
  glGenVertexArrays(1, &this->VAO);
  glBindVertexArray(this->VAO);

  // vertex shader input 
  glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(2);

  glBindBuffer(GL_ARRAY_BUFFER, VBO[3]);
  glVertexAttribIPointer(3, 3, GL_INT, GL_FALSE, 3 * sizeof(int), (void*)0);
  glEnableVertexAttribArray(3);

  glBindVertexArray(0);
}


void OffscreenGL::camera_setup(
  float zNear, float zFar, float *intrinsics, 
  unsigned int imgHeight, unsigned int imgWidth,
  float *camera_position, int index) {
  float fcv[] = {intrinsics[0], intrinsics[1]};
  float ccv[] = {intrinsics[2], intrinsics[3]};

  glm::vec3 cam_pos(camera_position[index * 3], camera_position[index * 3 + 1], camera_position[index * 3 + 2]); 
  glm::vec3 center(0.,0.,0.);
  glm::vec3 up(0.,1.,0.);
  glm::mat4 view_mat = glm::lookAt(cam_pos, center, up);

  glm::mat3 rot = glm::mat3(view_mat);
  glm::mat4 view_rotation_mat = glm::mat4(rot); // inverse transpose should be the same as rotation mat

  float left = - ccv[0] / fcv[0] * zNear;
  float bottom = (ccv[1] - (float)(imgHeight)) / fcv[1] * zNear;
  float right = ((float)imgWidth - ccv[0]) / fcv[0] * zNear;
  float top = ccv[1] / fcv[1] * zNear;

  glm::mat4 projection_mat = glm::frustum(left, right, bottom, top, zNear, zFar);
  glm::mat4 world_to_clip_mat = projection_mat * view_mat;

  // bind the mats
  GLuint location = glGetUniformLocation(this->shader_program, "world_to_view_mat");
  glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(view_mat));

  location = glGetUniformLocation(this->shader_program, "world_to_view_mat_inverse_transpose");
  glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(view_rotation_mat));

  location = glGetUniformLocation(this->shader_program, "world_to_clip_mat");
  glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(world_to_clip_mat));

  glViewport(0, 0, imgWidth, imgHeight);
}

void OffscreenGL::draw(
  unsigned char *imageBuffer, float *depthBuffer, bool *maskBuffer,
  float *normalBuffer, float *vertexBuffer,
  unsigned int imgHeight, unsigned int imgWidth,
  double *zNearFarV, int index, bool use_color = true) {

  // bind VAO, FBO & draw
  glBindFramebuffer(GL_FRAMEBUFFER, this->FBO);
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

  glBindVertexArray(this->VAO);
  glDrawArrays(GL_TRIANGLES, 0, fM * 3);
  glFlush();

  // stop using VAO
  glBindVertexArray(0);

  // bug fix for Nvidia
  unsigned int paddedWidth = imgWidth % 4;
  if (paddedWidth != 0) paddedWidth = 4 - paddedWidth + imgWidth;
  else                  paddedWidth = imgWidth;

  // Read off of the depth buffer
  GLfloat *data_buffer_depth = (GLfloat *)malloc(paddedWidth * imgHeight * sizeof(GLfloat));
  glReadPixels(0, 0, paddedWidth, imgHeight, GL_DEPTH_COMPONENT, GL_FLOAT, data_buffer_depth); //

  // Read off the vertex buffer
  GLfloat *data_buffer_vertex = (GLfloat *)malloc(4 * paddedWidth * imgHeight * sizeof(GLfloat));
  glReadBuffer(GL_COLOR_ATTACHMENT0); // specify the vertex buffer
  glReadPixels(0, 0, paddedWidth, imgHeight, GL_RGBA, GL_FLOAT, data_buffer_vertex);

  // Read off the normal buffer
  GLfloat *data_buffer_normal = (GLfloat *)malloc(4 * paddedWidth * imgHeight * sizeof(GLfloat));
  glReadBuffer(GL_COLOR_ATTACHMENT1); // specify the normal buffer
  glReadPixels(0, 0, paddedWidth, imgHeight, GL_RGBA, GL_FLOAT, data_buffer_normal);

  // Read off of the color buffer
  GLubyte *data_buffer_rgb = (GLubyte *)malloc(4 * paddedWidth * imgHeight * sizeof(GLubyte));
  glReadBuffer(GL_COLOR_ATTACHMENT2); // specify the color buffer
  glReadPixels(0, 0, paddedWidth, imgHeight, GL_RGBA, GL_UNSIGNED_BYTE, data_buffer_rgb);

  // reorder the pixel data for the opengl to numpy conversion
  unsigned int npImgIndex = 0;
  unsigned int oglImageIndex = 0;


  // buffer offset 
  imageBuffer += imgWidth * imgHeight * 4 * index;
  depthBuffer += imgWidth * imgHeight * 1 * index;
  maskBuffer += imgWidth * imgHeight * 1 * index;
  normalBuffer += imgWidth * imgHeight * 4 * index;
  vertexBuffer += imgWidth * imgHeight * 4 * index;

  float n = zNearFarV[0];
  float f = zNearFarV[1];
  // numpy save mat: H * W * C
  // C order saving
  for (int i = 0; i < imgHeight; i++) {
    for (int j = 0; j < imgWidth; j++, npImgIndex++) {
      oglImageIndex = (j + (imgHeight-1-i) * paddedWidth);
      float depth = (float) data_buffer_depth[oglImageIndex];

      // render mask: indicating points inside the clipped plane
      maskBuffer[npImgIndex] = depth < 1;

      // render depth
      depthBuffer[npImgIndex] = -f*n/(depth*(f-n)-f);

      // render normal
      normalBuffer[npImgIndex*4] = (float) data_buffer_normal[oglImageIndex*4];
      normalBuffer[npImgIndex*4+1] = (float) data_buffer_normal[oglImageIndex*4+1];
      normalBuffer[npImgIndex*4+2] = (float) data_buffer_normal[oglImageIndex*4+2];
      normalBuffer[npImgIndex*4+3] = (float) data_buffer_normal[oglImageIndex*4+3];
      
      // render vertex
      vertexBuffer[npImgIndex*4] = (float) data_buffer_vertex[oglImageIndex*4];
      vertexBuffer[npImgIndex*4+1] = (float) data_buffer_vertex[oglImageIndex*4+1];
      vertexBuffer[npImgIndex*4+2] = (float) data_buffer_vertex[oglImageIndex*4+2];
      vertexBuffer[npImgIndex*4+3] = (float) data_buffer_vertex[oglImageIndex*4+3];

      // render color
      if (use_color) {
        imageBuffer[npImgIndex*4] = (unsigned char) data_buffer_rgb[oglImageIndex*4];
        imageBuffer[npImgIndex*4+1] = (unsigned char) data_buffer_rgb[oglImageIndex*4+1];
        imageBuffer[npImgIndex*4+2] = (unsigned char) data_buffer_rgb[oglImageIndex*4+2];
        imageBuffer[npImgIndex*4+3] = (unsigned char) data_buffer_rgb[oglImageIndex*4+3];
      }
    }
  }


  free(data_buffer_depth);
  free(data_buffer_vertex);
  free(data_buffer_normal);
  free(data_buffer_rgb);

  // unbind FBO
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

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
  ) {
  //createGLContext();
  OffscreenGL ogl(imgSizeV[0], imgSizeV[1]);
  bool flag = ogl.initialize_shaders();

  if (!flag) {
    printf("Invalid shaders!");
    return;
  }

  ogl.prepare_vertex_info(vertex_array, color_array, normal_array, fM, use_color);
  for (int i=0;i<T;i++) {
    ogl.camera_setup(zNearFarV[0], zNearFarV[1], intrinsics, imgSizeV[0], imgSizeV[1], camera_position, i);
    ogl.draw(imageBuffer, depthBuffer, maskBuffer, normalBuffer, vertexBuffer,
      imgSizeV[0], imgSizeV[1], zNearFarV, i, use_color);
  }
}
