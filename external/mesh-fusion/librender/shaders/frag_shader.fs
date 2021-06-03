#version 130
flat in int face_id;
in vec3 vertex_color;
in vec4 position_world;
in vec4 position_camera;
in vec3 normal_camera;
in vec3 normal_world;

out vec4 frag_world_xyz;
out vec4 frag_normal;
out vec4 frag_color;

void main(){
    // view vector in
    vec3 view_vector = -position_camera.xyz;
    // Check if we need to flip the normal.
    vec3 normal_world_cor; // = normal_world;
    float d = dot(normalize(normal_camera), normalize(view_vector));
    //if (abs(d) < 0.001) {
        // we consider this view not to be a good observation
    //    frag_world_xyz = vec4(0,0,0,0);
    //    frag_normal = vec4(0,0,0,0);
    //    frag_color = vec4(0,0,0,0);
    //}
    //else{
        if (d < 0) {
            normal_world_cor = -normal_world;
        } else {
            normal_world_cor = normal_world;
        }

        float indicator = 0;
        if (face_id >= 0) 
            indicator = float(face_id) + 1.0f;

        frag_world_xyz = position_world;
        frag_world_xyz.w = indicator;
        frag_normal = vec4(normalize(normal_world_cor), indicator);
        frag_color = vec4(vertex_color, 1.0);
    //}
    //gl_FragColor = frag_normal;
}