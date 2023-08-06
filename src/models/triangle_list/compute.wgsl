@group(0) @binding(0)
var color_output: texture_storage_2d<rgba8unorm, write>; // this is used as both input and output for convenience

@group(0) @binding(1)
var depth_input: texture_2d<f32>; // this is used as both input and output for convenience

@group(0) @binding(2)
var depth_output: texture_storage_2d<r32float, write>; // this is used as both input and output for convenience

struct Camera {
    viewmodel_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    origin: vec3<f32>,
}

struct Screen {
    width: u32,
    height: u32,
}

struct Triangle {
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct HitRecord {
    hit: bool,
    distance: f32,
    normal: vec3<f32>,
}

var<private> kNoHit : HitRecord = HitRecord(false, 0.0f, vec3<f32>(0.0f, 0.0f, 0.0f));
var<private> kNear : f32 = 0.01;
var<private> kFar : f32 = 100.0;
var<private> kEpsilon : f32 = 0.000001;

var<private> kLightDir : vec3<f32> = vec3<f32>(1.0f, -5.0f, 1.0f);

@group(0) @binding(3)
var<uniform> camera: Camera;

@group(0) @binding(4)
var<uniform> screen: Screen;

@group(0) @binding(5)
var<storage> triangle_list: array<Triangle>;

fn buildHitRecordAtIntersection(ray: Ray, center: vec3<f32>, intersection_t: f32) -> HitRecord {
    let intersection_point = ray.origin + ray.direction * intersection_t;
    let normal = normalize(intersection_point - center);

    return HitRecord(true, intersection_t, normal);
}

fn toNonLinearDepth(depth: f32) -> f32 {
    return ((1.0 / depth) - (1.0 / kNear)) / ((1.0 / kFar) - (1.0 / kNear));
}

fn triangleRayIntersect(triangle: Triangle, ray: Ray) -> HitRecord {
    // compute the plane's normal
    let v0v1 = triangle.p1 - triangle.p0;
    let v0v2 = triangle.p2 - triangle.p0;

    var N = cross(v0v1, v0v2); // N
 
    // Step 1: finding P
    
    // check if the ray and plane are parallel.
    let NdotRayDirection = dot(N, ray.direction);
    if (abs(NdotRayDirection) < kEpsilon) {// almost 0
        return kNoHit; // they are parallel, so they don't intersect! 
    }

    // compute d parameter using equation 2
    let d = -dot(N, triangle.p0);
    
    // compute t (equation 3)
    let t = -(dot(N, ray.origin) + d) / NdotRayDirection;
    
    // check if the triangle is behind the ray
    if (t < 0.0f) {
        return kNoHit; // the triangle is behind
    }

    // compute the intersection point using equation 1
    let P = ray.origin + t * ray.direction;
 
    // Step 2: inside-outside test
    
    // edge 0
    let edge0 = triangle.p1 - triangle.p0; 
    let vp0 = P - triangle.p0;
    var C = cross(edge0, vp0);
    if (dot(N, C) < 0.0f) {
        return kNoHit; // P is on the right side
    }

    // edge 1
    let edge1 = triangle.p2 - triangle.p1; 
    let vp1 = P - triangle.p1;
    C = cross(edge1, vp1);
    if (dot(N, C) < 0.0f) {
        return kNoHit; // P is on the right side
    }

    // edge 2
    let edge2 = triangle.p0 - triangle.p2; 
    let vp2 = P - triangle.p2;
    C = cross(edge2, vp2);
    if (dot(N, C) < 0.0f) {
        return kNoHit; // P is on the right side;
    }

    if (NdotRayDirection > 0.0f) {
        N = -N;
    }

    return HitRecord(true, t, N);
}

fn pixelToRay(x: u32, y: u32) -> Ray {
    let x_nds = 2.0f * (f32(x) + 0.5f) / f32(screen.width) - 1.0f;
    let y_nds = 2.0f * (f32(y) + 0.5f) / f32(screen.height) - 1.0f;

    let proj_vec = vec4<f32>(x_nds, y_nds, 1.0f, 1.0f);
    var view_vec = camera.proj_inv * proj_vec;

    view_vec.w = 0.0f;

    let world_vec = camera.viewmodel_inv * view_vec;

    let ray_dir = normalize(world_vec.xyz);

    return Ray(camera.origin, ray_dir);
}

fn pixelToRay_ortho(x: u32, y: u32) -> Ray {
    let x_nds = 2.0f * (f32(x) + 0.5f) / f32(screen.width) - 1.0f;
    let y_nds = 2.0f * (f32(y) + 0.5f) / f32(screen.height) - 1.0f;

    let ray_origin = camera.origin + vec3<f32>(x_nds * 5.0f, y_nds * 5.0f, 0.0f);
    let ray_dir = vec3<f32>(0.0f, 0.0f, -1.0f);

    return Ray(ray_origin, ray_dir);
}


@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // if (global_id.x >= screen.width || global_id.y >= screen.height || global_id.z >= 1u) {
    //     return;
    // }

    let ray = pixelToRay(global_id.x, global_id.y);

    var i_min = 0;
    var min_hit = kNoHit;
    let element_count = arrayLength(&triangle_list);

    for (var i = u32(0); i < element_count; i++) {
        let triangle = triangle_list[i];
        let hit_record = triangleRayIntersect(triangle, ray);

        if ((!min_hit.hit && hit_record.hit) || (hit_record.hit && hit_record.distance < min_hit.distance)) {
            min_hit = hit_record;
        }
    }

    let hit_record = min_hit;

    var final_color = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);

    if (hit_record.hit) {
        // Custom depth testing
        let current_depth = 1.0 - textureLoad(depth_input, global_id.xy, 0).r;
        let depth = toNonLinearDepth(hit_record.distance);

        if (depth >= current_depth) {
            return;
        }

        // Shading
        let ambiant_comp = 0.1f;
        let diffuse_comp = 1.0f;
        let specular_comp = 0.5f;

        let diffuse = diffuse_comp * max(0.0f, dot(hit_record.normal, -normalize(kLightDir)));

        let half_dir = normalize(-normalize(kLightDir) - ray.direction);
        let specular = specular_comp * pow(max(0.0f, dot(half_dir, hit_record.normal)), 32.0f);

        let mat_color = vec3<f32>(1.0f, 0.0f, 0.0f);

        let diffuse_color = vec4<f32>((ambiant_comp + diffuse) * mat_color, 1.0f);
        let specular_color = vec4<f32>(specular * vec3<f32>(1.0f), 1.0f);

        final_color = diffuse_color + specular_color;
        //

        textureStore(depth_output, vec2<u32>(global_id.x, global_id.y), vec4<f32>(1.0 - depth, 0.0, 0.0, 0.0));
        textureStore(color_output, vec2<u32>(global_id.x, global_id.y), final_color);
    }
}
