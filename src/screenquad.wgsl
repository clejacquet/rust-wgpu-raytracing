// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}
struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;


fn nearest(i: vec2<i32>) -> vec4<f32> {
    return textureLoad(t_diffuse, i, 0);
}

fn linear(p: vec2<f32>, texture_size: vec2<u32>) -> vec4<f32> {
    let left = clamp(i32(floor(p.x)), 0, i32(texture_size.x) - 1);
    let right = clamp(i32(ceil(p.x)), 0, i32(texture_size.x) - 1);
    let top = clamp(i32(floor(p.y)), 0, i32(texture_size.y) - 1);
    let bottom = clamp(i32(ceil(p.y)), 0, i32(texture_size.y) - 1);

    let v_lt = textureLoad(t_diffuse, vec2<i32>(left, top), 0);
    let v_rt = textureLoad(t_diffuse, vec2<i32>(right, top), 0);
    let v_lb = textureLoad(t_diffuse, vec2<i32>(left, bottom), 0);
    let v_rb = textureLoad(t_diffuse, vec2<i32>(right, bottom), 0);

    let dx = p.x - floor(p.x);
    let dy = p.y - floor(p.y);

    let top_val = mix(v_lt, v_rt, dx);
    let bottom_val = mix(v_lb, v_rb, dx);
    let final_val = mix(top_val, bottom_val, dy);

    return final_val;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_size = textureDimensions(t_diffuse);
    let p_x = f32(in.tex_coords.x) * f32(texture_size.x) - 0.5f;
    let p_y = f32(in.tex_coords.y) * f32(texture_size.y) - 0.5f;

    let i_x = clamp(i32(round(p_x)), 0, i32(texture_size.x) - 1);
    let i_y = clamp(i32(round(p_y)), 0, i32(texture_size.y) - 1);

    // return nearest(vec2<i32>(i_x, i_y));
    // return linear(vec2<f32>(p_x, p_y), texture_size);

    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
    // return vec4<f32>(in.tex_coords, 0.0, 1.0);
}