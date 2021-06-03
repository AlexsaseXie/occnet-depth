#include "offscreen_new.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

int OffscreenGL_New::glutWin = -1;
bool OffscreenGL_New::glutInitialized = false;

OffscreenGL_New::OffscreenGL_New(int maxHeight, int maxWidth) {

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

    glGenRenderbuffers(1, &this->rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, this->rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, maxWidth, maxHeight); // use a single renderbuffer object for both a depth AND stencil buffer.
    
    glGenTextures(1, &this->normal_tex);
    glBindTexture(GL_TEXTURE_2D, this->normal_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, maxWidth, maxHeight, 0, GL_RGBA, GL_FLOAT, 0);

    glGenTextures(1, &this->vertex_tex);
    glBindTexture(GL_TEXTURE_2D, this->vertex_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, maxWidth, maxHeight, 0, GL_RGBA, GL_FLOAT, 0);
    
    // FBO
    glGenFramebuffers(1, &this->FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, this->FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->vertex_tex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, this->normal_tex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, this->render_tex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, this->rbo); // now actually attach it
     
    GLenum draw_buffers[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    
    glDrawBuffers(3, draw_buffers);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  } else {
    // WARNING: shouldn't be called
    glutSetWindow(glutWin);
  }
}

OffscreenGL_New::~OffscreenGL_New() {
  // free space
  if (this->delete_color_flag)
    delete [] this->t_color_array;
  
  delete [] this->idx_array;
}

// load shaders
static const GLchar* getFileData( const char * path ) {
	ifstream infile(path);

	infile.seekg(0, infile.end);
	int len = infile.tellg();
	infile.seekg(0, infile.beg);

	GLchar *source = new GLchar[len + 1];
	infile.read(source, len);
	infile.close();
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
		printf("%s\n", &ShaderErrorMessage[0]);
		return false;
	}

	return true;
}

void checkCompileErrors(GLuint shader, const char * type) {
  GLint success;
  GLchar infoLog[1024];
  if(type != "PROGRAM")
  {
      glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
      if(!success)
      {
          glGetShaderInfoLog(shader, 1024, NULL, infoLog);
          std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
      }
  }
  else
  {
      glGetProgramiv(shader, GL_LINK_STATUS, &success);
      if(!success)
      {
          glGetProgramInfoLog(shader, 1024, NULL, infoLog);
          std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
      }
  }
}

bool OffscreenGL_New::initialize_shaders() {
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
  checkCompileErrors(this->shader_program, "PROGRAM");
  glUseProgram(this->shader_program);

  return true;
}

// vertex info : VAO & VBO
void OffscreenGL_New::prepare_vertex_info(
  float *vertex_array, float *color_array, float *normal_array, int fM, // 3 * N array
  bool use_color) {
  this->face_count = fM; // has fM * 3 vertices
  // color
  if (!use_color) {
    this->t_color_array = new float [fM * 3 * 3];
    for (int i=0; i < fM * 9; i += 3) {
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
  glGenBuffers(4, this->VBO);

  glBindBuffer(GL_ARRAY_BUFFER, this->VBO[0]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * fM * 9, vertex_array, GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, this->VBO[1]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * fM * 9, normal_array, GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, this->VBO[2]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * fM * 9, this->t_color_array, GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, this->VBO[3]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(int) * fM * 3, this->idx_array, GL_STATIC_DRAW);

  // VAO
  glGenVertexArrays(1, &this->VAO);
  glBindVertexArray(this->VAO);

  // vertex shader input 
  glBindBuffer(GL_ARRAY_BUFFER, this->VBO[0]);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
  

  glBindBuffer(GL_ARRAY_BUFFER, this->VBO[1]);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
  

  glBindBuffer(GL_ARRAY_BUFFER, this->VBO[2]);
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
 

  glBindBuffer(GL_ARRAY_BUFFER, this->VBO[3]);
  glEnableVertexAttribArray(3);
  glVertexAttribIPointer(3, 1, GL_INT, 0, 0);
  
  glBindVertexArray(0);
}

void OffscreenGL_New::camera_setup(
  float zNear, float zFar, float *intrinsics, 
  unsigned int imgHeight, unsigned int imgWidth,
  float *camera_position, int index) {
  float fcv[] = {intrinsics[0], intrinsics[1]};
  float ccv[] = {intrinsics[2], intrinsics[3]};

  glm::vec3 cam_pos(camera_position[index * 3], camera_position[index * 3 + 1], camera_position[index * 3 + 2]); 
  glm::vec3 center(0,0,0);
  glm::vec3 up(0,1,0);
  this->view_mat = glm::lookAt(cam_pos, center, up);

  //printf("Camera position: (%f, %f, %f)\n", camera_position[index * 3], camera_position[index * 3+1], camera_position[index * 3 + 2]);

  glm::mat3 rot = glm::mat3(this->view_mat);
  glm::mat3 view_rotation_mat = rot; // inverse transpose should be the same as rotation mat

  float left = - ccv[0] / fcv[0] * zNear;
  float bottom = (ccv[1] - (float)(imgHeight)) / fcv[1] * zNear;
  float right = ((float)imgWidth - ccv[0]) / fcv[0] * zNear;
  float top = ccv[1] / fcv[1] * zNear;

  glm::mat4 projection_mat = glm::frustum(left, right, bottom, top, zNear, zFar);
  glm::mat4 world_to_clip_mat = projection_mat * this->view_mat;

  // bind the mats
  GLint location = glGetUniformLocation(this->shader_program, "world_to_view_mat");
  glUniformMatrix4fv(location, 1, GL_FALSE, &this->view_mat[0][0]);

  location = glGetUniformLocation(this->shader_program, "world_to_view_mat_inverse_transpose");
  glUniformMatrix3fv(location, 1, GL_FALSE, &view_rotation_mat[0][0]);

  location = glGetUniformLocation(this->shader_program, "world_to_clip_mat");
  glUniformMatrix4fv(location, 1, GL_FALSE, &world_to_clip_mat[0][0]);

  glViewport(0, 0, imgWidth, imgHeight);
}

void OffscreenGL_New::draw(
  unsigned char *imageBuffer, float *depthBuffer, bool *maskBuffer,
  float *normalBuffer, float *vertexBuffer,
  unsigned int imgHeight, unsigned int imgWidth,
  float *zNearFarV, int index, bool use_color) {

  // bind VAO, FBO & draw
  glBindFramebuffer(GL_FRAMEBUFFER, this->FBO);
  glEnable(GL_DEPTH_TEST);
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

  glUseProgram(this->shader_program);
  glBindVertexArray(this->VAO);
  glDrawArrays(GL_TRIANGLES, 0, this->face_count * 3);

  //glFlush();

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
  // unbind VAO
  glBindVertexArray(0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void OffscreenGL_New::copy_view_mat(float * viewMatBuffer, int index) {
  viewMatBuffer += 16 * index;
  for (int j=0;j<4;j++) {
    for (int k=0;k<4;k++) {
      *(viewMatBuffer + (j*4) + k) = (float)this->view_mat[j][k];
    }
  }
}

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
  ) {
  //createGLContext();
  OffscreenGL_New ogl(imgSizeV[0], imgSizeV[1]);
  bool flag = ogl.initialize_shaders();

  ogl.prepare_vertex_info(vertex_array, color_array, normal_array, fM, use_color);
  //GLenum a = glGetError();
  //printf("After prepare vertex info %u\n", a);
  for (int i=0;i<T;i++) {
    ogl.camera_setup(zNearFarV[0], zNearFarV[1], intrinsics, imgSizeV[0], imgSizeV[1], camera_position, i);
    //a = glGetError();
    //printf("After camera setup %u\n", a);
    ogl.draw(imageBuffer, depthBuffer, maskBuffer, normalBuffer, vertexBuffer,
      imgSizeV[0], imgSizeV[1], zNearFarV, i, use_color);
    //a = glGetError();
    //printf("After draw %u\n", a);
    ogl.copy_view_mat(viewMatBuffer, i);
  }
}

void select_vertex(  // input
  float * vertexBuffer, float * normalBuffer,
  int * imgSizeV,
  int fM, int T,
  // output
  float *point_cloud, int *point_cloud_size,
  int *stats
) {
  int H = imgSizeV[0];
  int W = imgSizeV[1];

  glm::vec3 * face_normal = new glm::vec3[fM];
  for (int i=0;i<fM;i++) {
    face_normal[i] = glm::vec3(0,0,0);
  }

  int ** face_normal_pixel_count = new int * [fM]; // [0] positive, [1] negative
  for (int i=0;i<fM;i++) {
    face_normal_pixel_count[i] = new int[2];
    for (int j=0;j<2;j++) {
      face_normal_pixel_count[i][j] = 0;
    }
  }

  int total_visible_face_count = 0;
  int double_sided_face_count = 0;
  int bad_face_count = 0;

  int offset = 4;
  float * normalBufferPointer = normalBuffer;
  int total_pixel_count = T * H * W;
  // decide the normal
  //for (int t=0;t<T;t++) {
  //  for (int y=0;y<H;y++) {
  //    for (int x=0;x<W;x++) {
  for (int i=0; i < total_pixel_count; i++) {
    normalBufferPointer += offset;
    int face_id = static_cast<int>(*(normalBufferPointer + 3) + 0.1) - 1;
    if (face_id >= 0) {
      glm::vec3 cur_normal = glm::vec3(*(normalBufferPointer), *(normalBufferPointer + 1), *(normalBufferPointer + 2));
      if (face_normal[face_id] == glm::vec3(0,0,0)) {
        face_normal[face_id] = cur_normal;
        face_normal_pixel_count[face_id][0] = 1;

        total_visible_face_count ++;
      }
      else {
        bool cur_positive = glm::dot(face_normal[face_id], cur_normal) >= 0; // face_normal[face_id] == cur_normal;
        face_normal_pixel_count[face_id][cur_positive ? 0 : 1] ++;
      }
    }
  }
  //    }
  //  }
  //}

  //flip if needed
  for (int i=0;i<fM;i++) {
    if (face_normal_pixel_count[i][0] > 0) {
      if (face_normal[i][1] > 0) 
        double_sided_face_count ++;

      if (face_normal_pixel_count[i][0] < face_normal_pixel_count[i][1]) {
        face_normal[0] = -face_normal[0];
        face_normal[1] = -face_normal[1];
        face_normal[2] = -face_normal[2];
        int tmp = face_normal_pixel_count[i][0];
        face_normal_pixel_count[i][0] = face_normal_pixel_count[i][1];
        face_normal_pixel_count[i][1] = tmp;
      }

      if (static_cast<float>(face_normal_pixel_count[i][1]) >= static_cast<float>(face_normal_pixel_count[i][0]) * 0.1f)
        bad_face_count ++;
    }
  }

  //printf("double-sided faces: %d, bad faces: %d, total faces:%d\n", double_sided_face_count, bad_face_count, total_visible_face_count);

  // insert vertex
  float * vertexBufferPointer = vertexBuffer;
  normalBufferPointer = normalBuffer;

  int max_pointcloud_size = *point_cloud_size;
  int point_cloud_count = 0;
  float * current_head = point_cloud;
  for (int i=0;i<total_pixel_count;i++) {
    vertexBufferPointer += offset;
    normalBufferPointer += offset;
    int face_id = static_cast<int>(*(normalBufferPointer + 3) + 0.1) - 1;
    if (face_id >= 0) {
      glm::vec3 cur_normal = glm::vec3(*(normalBufferPointer), *(normalBufferPointer + 1), *(normalBufferPointer + 2));
      bool cur_positive = glm::dot(face_normal[face_id], cur_normal) >= 0; // face_normal[face_id] == cur_normal;
      if (cur_positive) {          
        *(current_head) = *(vertexBufferPointer);
        *(current_head + 1) = *(vertexBufferPointer + 1);
        *(current_head + 2) = *(vertexBufferPointer + 2);
        *(current_head + 3) = cur_normal[0];
        *(current_head + 4) = cur_normal[1];
        *(current_head + 5) = cur_normal[2];

        current_head += 6;
        point_cloud_count ++;
        if (point_cloud_count >= max_pointcloud_size) break;
      }
    }
  }

  *point_cloud_size = point_cloud_count;
  printf("point cloud size: %d\n", point_cloud_count);
  stats[0] = double_sided_face_count;
  stats[1] = bad_face_count;
  stats[2] = total_visible_face_count;

  // free space
  delete [] face_normal;
  for (int i=0;i<fM;i++) {
    delete [] face_normal_pixel_count[i];
  }
  delete [] face_normal_pixel_count;
}
