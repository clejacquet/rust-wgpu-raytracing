@group(0) @binding(0)
var output: texture_storage_2d<rgba8unorm,write>; // this is used as both input and output for convenience

struct Camera {
    viewmodel_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    origin: vec3<f32>,
}

struct Screen {
    width: u32,
    height: u32,
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

@group(0) @binding(1)
var<uniform> camera: Camera;

@group(0) @binding(2)
var<uniform> screen: Screen;

fn buildHitRecordAtIntersection(ray: Ray, center: vec3<f32>, intersection_t: f32) -> HitRecord {
    let intersection_point = ray.origin + ray.direction * intersection_t;
    let normal = normalize(intersection_point - center);

    return HitRecord(true, intersection_t, normal);
}

fn sphereRayIntersect(center: vec3<f32>, radius: f32, ray: Ray) -> HitRecord {
    let oc = ray.origin - center;
    let a = dot(ray.direction, ray.direction);
    let b = 2.0 * dot(oc, ray.direction);
    let c = dot(oc, oc) - (radius * radius);

    let discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) {
        return kNoHit;
    }

    let sqrtDiscriminant = sqrt(discriminant);
    let t1 = (-b - sqrtDiscriminant) / (2.0 * a);
    let t2 = (-b + sqrtDiscriminant) / (2.0 * a);

    if (t1 >= 0.0) {
        return buildHitRecordAtIntersection(ray, center, t1);
    } else if (t2 >= 0.0) {
        return buildHitRecordAtIntersection(ray, center, t2);
    } else {
        return kNoHit; // Both intersection points are behind the ray's origin
    }
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
    if (global_id.x >= screen.width || global_id.y >= screen.height || global_id.z >= 1u) {
        return;
    }

    let ray = pixelToRay(global_id.x, global_id.y);

    let hit_record = sphereRayIntersect(vec3<f32>(1.0f, 0.5f, -3.0f), 2.0f, ray);

    var final_color = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);

    if (hit_record.hit) {
        let diffuse = abs(dot(hit_record.normal, ray.direction));

        final_color = vec4<f32>(diffuse, 0.0f, 0.0f, 1.0f);
    }

    textureStore(output, vec2<u32>(global_id.x, global_id.y), final_color);
}
