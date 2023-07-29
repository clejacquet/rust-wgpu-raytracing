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

fn _cubic_base(p: f32, v_0: vec4<f32>, v_1: vec4<f32>, v_2: vec4<f32>, v_3: vec4<f32>) -> vec4<f32> {
    let f0 = v_1;
    let f1 = v_2;
    let d0 = (v_2 - v_0) / 2.0f;
    let d1 = (v_3 - v_1) / 2.0f;

    let a = 2.0f * f0 - 2.0f * f1 + d0 + d1;
    let b = -3.0f * f0 + 3.0f * f1 - 2.0f * d0 - d1;
    let c = d0;
    let d = f0;

    let x = p - floor(p);

    return a * pow(x, 3.0f) + b * pow(x, 2.0f) + c * x + d;
}

fn _cubic1dx(p: f32, i_y: i32, texture_size: vec2<u32>) -> vec4<f32> {
    let i_x0 = clamp(i32(floor(p)) - 1, 0, i32(texture_size.x) - 1);
    let i_x1 = clamp(i32(floor(p)), 0, i32(texture_size.x) - 1);
    let i_x2 = clamp(i32(ceil(p)), 0, i32(texture_size.x) - 1);
    let i_x3 = clamp(i32(ceil(p)) + 1, 0, i32(texture_size.x) - 1);

    let v_0 = textureLoad(t_diffuse, vec2<i32>(i_x0, i_y), 0);
    let v_1 = textureLoad(t_diffuse, vec2<i32>(i_x1, i_y), 0);
    let v_2 = textureLoad(t_diffuse, vec2<i32>(i_x2, i_y), 0);
    let v_3 = textureLoad(t_diffuse, vec2<i32>(i_x3, i_y), 0);

    return _cubic_base(p - f32(i_x1), v_0, v_1, v_2, v_3);
} 

fn cubic(p: vec2<f32>, texture_size: vec2<u32>) -> vec4<f32> {
    let i_y0 = clamp(i32(floor(p.y)) - 1, 0, i32(texture_size.y) - 1);
    let i_y1 = clamp(i32(floor(p.y)), 0, i32(texture_size.y) - 1);
    let i_y2 = clamp(i32(ceil(p.y)), 0, i32(texture_size.y) - 1);
    let i_y3 = clamp(i32(ceil(p.y)) + 1, 0, i32(texture_size.y) - 1);

    let v_0 = _cubic1dx(p.x, i_y0, texture_size);
    let v_1 = _cubic1dx(p.x, i_y1, texture_size);
    let v_2 = _cubic1dx(p.x, i_y2, texture_size);
    let v_3 = _cubic1dx(p.x, i_y3, texture_size);

    return _cubic_base(p.y - f32(i_y1), v_0, v_1, v_2, v_3);
}

fn _cubic_fast_base(v: f32) -> vec4<f32> {
    let n = vec4<f32>(1.0, 2.0, 3.0, 4.0) - v;
    let s = n * n * n;
    let x = s.x;
    let y = s.y - 4.0f * s.x;
    let z = s.z - 4.0f * s.y + 6.0f * s.x;
    let w = 6.0f - x - y - z;
    return vec4<f32>(x, y, z, w) * (1.0f / 6.0f);
}

fn cubic_fast(p: vec2<f32>, texture_size: vec2<u32>) -> vec4<f32>{
    let invTexSize = 1.0f / vec2<f32>(texture_size);

    let fxy = fract(p);
    var p = p - fxy;

    let xcubic = _cubic_fast_base(fxy.x);
    let ycubic = _cubic_fast_base(fxy.y);

    let c = p.xxyy + vec2<f32>(-0.5f, 1.5f).xyxy;
    
    let s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    var offset = c + vec4<f32>(xcubic.yw, ycubic.yw) / s;
    
    offset *= invTexSize.xxyy;
    
    let sample0 = textureSample(t_diffuse, s_diffuse, offset.xz);
    let sample1 = textureSample(t_diffuse, s_diffuse, offset.yz);
    let sample2 = textureSample(t_diffuse, s_diffuse, offset.xw);
    let sample3 = textureSample(t_diffuse, s_diffuse, offset.yw);

    let sx = s.x / (s.x + s.y);
    let sy = s.z / (s.z + s.w);

    return mix(
       mix(sample3, sample2, sx), mix(sample1, sample0, sx)
    , sy);
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
    return cubic(vec2<f32>(p_x, p_y), texture_size);
    // return cubic_fast(vec2<f32>(p_x, p_y), texture_size);

    // return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}