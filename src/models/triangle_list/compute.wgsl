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

struct VertexInfo {
    pos: vec3<f32>,
    tex: vec2<f32>,
}

struct Material {
    ambient: vec3<f32>,
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct HitRecord {
    hit: bool,
    distance: f32,
    normal: vec3<f32>,
    barycentric: vec3<f32>,
}

struct HitResult {
    record: HitRecord,
    object_id: u32,
}

struct BvhNodeData {
    left_child: i32,
    right_child: i32,
    first_prim: u32,
    prim_count: u32,
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
}

struct BvhMetaData {
    root_id: u32,
    node_count: u32,
}

var<private> kNoHit : HitRecord = HitRecord(false, 0.0f, vec3<f32>(0.0f, 0.0f, 0.0f), vec3<f32>(0.0f, 0.0f, 0.0f));
var<private> kNoHitResult : HitResult = HitResult(HitRecord(false, 0.0f, vec3<f32>(0.0f, 0.0f, 0.0f), vec3<f32>(0.0f, 0.0f, 0.0f)), 0u);
var<private> kNear : f32 = 0.01;
var<private> kFar : f32 = 100.0;
var<private> kEpsilon : f32 = 0.000001;

var<private> kLightDir : vec3<f32> = vec3<f32>(1.0f, -1.0f, -5.0f);

@group(0) @binding(3)
var<uniform> camera: Camera;

@group(0) @binding(4)
var<uniform> screen: Screen;

@group(0) @binding(5)
var<storage> vertice_list: array<VertexInfo>;

@group(0) @binding(6)
var<storage> face_list: array<vec3<u32>>;

@group(0) @binding(7)
var<uniform> material: Material;

@group(0) @binding(8)
var<storage> bvh_nodes: array<BvhNodeData>;

@group(0) @binding(9)
var<uniform> bvh_metadata: BvhMetaData;

@group(1) @binding(0)
var texture_diffuse: texture_2d<f32>;

@group(1) @binding(1)
var sampler_diffuse: sampler;

fn toNonLinearDepth(depth: f32) -> f32 {
    return ((1.0 / depth) - (1.0 / kNear)) / ((1.0 / kFar) - (1.0 / kNear));
}

fn triangleRayIntersect(triangle: Triangle, ray: Ray) -> HitRecord {
    // compute the plane's normal
    let v0v1 = triangle.p1 - triangle.p0;
    let v0v2 = triangle.p2 - triangle.p0;

    var N = cross(v0v1, v0v2); // N
    let denom = dot(N, N);
 
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
    var u = dot(N, C);
    if (u < 0.0f) {
        return kNoHit; // P is on the right side
    }

    // edge 2
    let edge2 = triangle.p0 - triangle.p2; 
    let vp2 = P - triangle.p2;
    C = cross(edge2, vp2);
    var v = dot(N, C);
    if (v < 0.0f) {
        return kNoHit; // P is on the right side;
    }

    if (NdotRayDirection > 0.0f) {
        N = -N;
    }

    u = u / denom;
    v = v / denom;

    return HitRecord(true, t, normalize(N), vec3<f32>(u, v, 1.0f - u - v));
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

fn scene_traversal_naive(ray: Ray) -> HitResult {
    var i_min = 0;
    var min_hit = kNoHit;
    let element_count = arrayLength(&face_list);

    for (var i = 0; i < i32(element_count); i++) {
        let vertex0 = vertice_list[face_list[i][0]].pos;
        let vertex1 = vertice_list[face_list[i][1]].pos;
        let vertex2 = vertice_list[face_list[i][2]].pos;
    
        let triangle = Triangle(vertex0, vertex1, vertex2);
        let hit_record = triangleRayIntersect(triangle, ray);

        if ((!min_hit.hit && hit_record.hit) || (hit_record.hit && hit_record.distance < min_hit.distance)) {
            min_hit = hit_record;
            i_min = i;
        }
    }

    return HitResult(min_hit, u32(i_min));
}

fn intersect_aabb(ray: Ray, bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    let tmin = (bmin - ray.origin) / ray.direction;
    let tmax = (bmax - ray.origin) / ray.direction;
    let t1 = min(tmin, tmax);
    let t2 = max(tmin, tmax);
    let tnear = max(max(t1.x, t1.y), t1.z);
    let tfar = min(min(t2.x, t2.y), t2.z);
    return tfar >= tnear;
};

fn intersect_bvh(ray: Ray, collision_list: ptr<function, array<i32, 128>>) -> i32 {
    var stack = array<i32, 64>();

    stack[0] = 0;
    var stack_counter = 1;

    var collision_counter = 0;

    for (var i = 0; i < 200 && stack_counter > 0 && stack_counter < 64; i++) {
        stack_counter -= 1;

        let node_id = stack[stack_counter];
        let node = bvh_nodes[node_id];

        let is_leaf = node.left_child == -1 && node.right_child == -1;
        let overlap = intersect_aabb(ray, node.aabb_min, node.aabb_max);

        if (overlap && is_leaf) {
            (*collision_list)[collision_counter] = node_id;
            collision_counter += 1;
        }

        if (overlap && !is_leaf) {
            stack[stack_counter] = node.left_child;
            stack_counter += 1;
            stack[stack_counter] = node.right_child;
            stack_counter += 1;
        }
    }

    return collision_counter;
}

fn scene_traversal_bvh(ray: Ray) -> HitResult {
    var collision_list = array<i32, 128>();
    let collision_count = intersect_bvh(ray, &collision_list);


    var i_min = 0;
    var min_hit = kNoHit;

    for (var c = 0; c < collision_count; c++) {
        let prim_start = i32(bvh_nodes[collision_list[c]].first_prim);
        let prim_end = prim_start + i32(bvh_nodes[collision_list[c]].prim_count);

        for (var i = prim_start; i < prim_end; i++) {
            let vertex0 = vertice_list[face_list[i][0]].pos;
            let vertex1 = vertice_list[face_list[i][1]].pos;
            let vertex2 = vertice_list[face_list[i][2]].pos;
        
            let triangle = Triangle(vertex0, vertex1, vertex2);
            let hit_record = triangleRayIntersect(triangle, ray);

            if ((!min_hit.hit && hit_record.hit) || (hit_record.hit && hit_record.distance < min_hit.distance)) {
                min_hit = hit_record;
                i_min = i;
            }
        }
    }

    return HitResult(min_hit, u32(i_min));
}


@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // if (global_id.x >= screen.width || global_id.y >= screen.height || global_id.z >= 1u) {
    //     return;
    // }

    let ray = pixelToRay(global_id.x, global_id.y);

    // let hit_result = scene_traversal_naive(ray);
    let hit_result = scene_traversal_bvh(ray);
    let hit_record = hit_result.record;

    var final_color = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);

    if (hit_record.hit) {
        // Custom depth testing
        let current_depth = 1.0 - textureLoad(depth_input, global_id.xy, 0).r;
        let depth = toNonLinearDepth(hit_record.distance);

        if (depth >= current_depth) {
            return;
        }

        // Shading
        let hit_object_id = hit_result.object_id;
        let tex_coords_0 = vertice_list[face_list[hit_object_id][0]].tex;
        let tex_coords_1 = vertice_list[face_list[hit_object_id][1]].tex;
        let tex_coords_2 = vertice_list[face_list[hit_object_id][2]].tex;
        var tex_coords = hit_record.barycentric[0] * tex_coords_0 + hit_record.barycentric[1] * tex_coords_1 + hit_record.barycentric[2] * tex_coords_2;        

        tex_coords = vec2<f32>(tex_coords.x, 1.0 - tex_coords.y);

        let tex_diffuse_val = textureSampleGrad(texture_diffuse, sampler_diffuse, tex_coords, vec2<f32>(0.0f, 0.0f), vec2<f32>(0.0f, 0.0f)).rgb;
        let diffuse = tex_diffuse_val * max(0.0f, dot(hit_record.normal, -normalize(kLightDir)));

        let half_dir = normalize(-normalize(kLightDir) - ray.direction);
        let specular = material.specular * pow(max(0.0f, dot(half_dir, hit_record.normal)), 32.0f);

        let diffuse_color = vec4<f32>(material.ambient + diffuse, 1.0f);
        let specular_color = vec4<f32>(specular * vec3<f32>(1.0f), 1.0f);

        final_color = diffuse_color + specular_color;
        //

        textureStore(depth_output, vec2<u32>(global_id.x, global_id.y), vec4<f32>(1.0 - depth, 0.0, 0.0, 0.0));
        textureStore(color_output, vec2<u32>(global_id.x, global_id.y), final_color);
    }
}
