#version 130
in vec3 vertex;
in vec3 normal;
in vec3 color;
in int id;

flat out int face_id;
out vec3 vertex_color;
out vec4 position_world;
out vec4 position_camera;
out vec3 normal_camera;
out vec3 normal_world;


//uniform mat4 projection_mat;
//uniform mat4 model_to_world_mat; // identity
uniform mat4 world_to_view_mat;
uniform mat3 world_to_view_mat_inverse_transpose;
uniform mat4 world_to_clip_mat; // projection * world_to_view * model_to_world

void main(){
    // Projected image coordinate
    gl_Position =  world_to_clip_mat * vec4(vertex,1);

    // world coordinate location of the vertex
    position_world = vec4(vertex, 1.0);
    position_camera = world_to_view_mat * position_world;

    // normal
    normal_world = normal;
    normal_camera = world_to_view_mat_inverse_transpose * normal;

    // color
    vertex_color = color;
    
    // id
    face_id = id;
}