// Vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
// here, group(1) refers to bind group 1, which is our camera binding group, 
// and binding(0) refers to the first (and only) binding within it, which is the camera uniform (within the camera buffer) 
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(
    point: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(point.position, 1.0);
    out.color = point.color;
    return out;
}



// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}


