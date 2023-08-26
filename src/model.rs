use std::ops::Range;

use crate::texture;

pub trait Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl ScreenVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ScreenVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertexSmall {
    pub position: [f32; 3],
    pub pad0: f32,
    pub tex_coords: [f32; 2],
    pub pad1: [f32; 2],
}

impl ModelVertexSmall {
    pub fn new(position: [f32; 3], tex_coords: [f32; 2]) -> Self {
        Self {
            position,
            pad0: 0.0,
            tex_coords,
            pad1: [0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelFaceSmall {
    pub indices: [u32; 3],
    pub pad0: u32,
}

impl ModelFaceSmall {
    pub fn new(indices: [u32; 3]) -> Self {
        Self { indices, pad0: 0 }
    }
}

impl Vertex for ModelVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub struct Material {
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
    pub ambient: cgmath::Vector3<f32>,
    pub diffuse: cgmath::Vector3<f32>,
    pub specular: cgmath::Vector3<f32>,
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub vertices: Vec<cgmath::Vector3<f32>>,
    pub triangle_indices: Vec<[usize; 3]>,
    pub num_elements: usize,
    pub material: usize,
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

#[derive(Clone)]
pub struct BvhNode {
    pub aabb_min: cgmath::Vector3<f32>,
    pub aabb_max: cgmath::Vector3<f32>,
    pub left_child: i32,
    pub right_child: i32,
    pub first_prim: usize,
    pub prim_count: usize,
}

pub struct BvhData {
    pub nodes: Vec<BvhNode>,
    pub root_id: usize,
    pub node_count: usize,
}

impl BvhData {
    pub fn new(primitive_count: usize) -> Self {
        Self {
            nodes: vec![
                BvhNode {
                    aabb_min: cgmath::Vector3::new(0.0, 0.0, 0.0),
                    aabb_max: cgmath::Vector3::new(0.0, 0.0, 0.0),
                    left_child: -1,
                    right_child: -1,
                    first_prim: 0,
                    prim_count: 0,
                };
                2 * primitive_count + 1
            ],
            root_id: 0,
            node_count: 0,
        }
    }

    fn update_bounds(
        &mut self,
        node_id: usize,
        triangles_indices: &Vec<[usize; 3]>,
        vertices: &Vec<cgmath::Vector3<f32>>,
    ) {
        let node = &self.nodes[node_id];

        if node.prim_count == 0 {
            let node_mut = &mut self.nodes[node_id];
            node_mut.aabb_min = cgmath::Vector3::new(0.0, 0.0, 0.0);
            node_mut.aabb_max = cgmath::Vector3::new(0.0, 0.0, 0.0);

            return;
        }

        let bounds = (node.first_prim..(node.first_prim + node.prim_count))
            .map(|prim_id| {
                let triangle = triangles_indices[prim_id].map(|vertex_id| vertices[vertex_id]);
                [
                    cgmath::Vector3::new(
                        triangle
                            .map(|vertex| vertex[0])
                            .iter()
                            .copied()
                            .min_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                        triangle
                            .map(|vertex| vertex[1])
                            .iter()
                            .copied()
                            .min_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                        triangle
                            .map(|vertex| vertex[2])
                            .iter()
                            .copied()
                            .min_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                    ),
                    cgmath::Vector3::new(
                        triangle
                            .map(|vertex| vertex[0])
                            .iter()
                            .copied()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                        triangle
                            .map(|vertex| vertex[1])
                            .iter()
                            .copied()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                        triangle
                            .map(|vertex| vertex[2])
                            .iter()
                            .copied()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                    ),
                ]
            })
            .reduce(|bound_acc, bound| {
                [
                    cgmath::Vector3::new(
                        [bound_acc[0][0], bound[0][0]]
                            .iter()
                            .copied()
                            .min_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                        [bound_acc[0][1], bound[0][1]]
                            .iter()
                            .copied()
                            .min_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                        [bound_acc[0][2], bound[0][2]]
                            .iter()
                            .copied()
                            .min_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                    ),
                    cgmath::Vector3::new(
                        [bound_acc[1][0], bound[1][0]]
                            .iter()
                            .copied()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                        [bound_acc[1][1], bound[1][1]]
                            .iter()
                            .copied()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                        [bound_acc[1][2], bound[1][2]]
                            .iter()
                            .copied()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                    ),
                ]
            });

        self.nodes[node_id].aabb_min = bounds.unwrap()[0];
        self.nodes[node_id].aabb_max = bounds.unwrap()[1];
    }
}

fn get_centroid(triangle: [cgmath::Vector3<f32>; 3]) -> cgmath::Vector3<f32> {
    0.5 * triangle[0] + 0.5 * triangle[1] + 0.5 * triangle[2]
}

fn subdivide_bvh(
    bvh_data: &mut BvhData,
    node_id: usize,
    triangles_indices: &mut Vec<[usize; 3]>,
    vertices: &Vec<cgmath::Vector3<f32>>,
) {
    let mut i = 0;
    let mut j = 0;
    {
        let node = &bvh_data.nodes[node_id];
        let extent = node.aabb_max - bvh_data.nodes[node_id].aabb_min;

        let aabb_min: [f32; 3] = node.aabb_min.into();
        let extent_slice: [f32; 3] = extent.into();

        let mut axis = 0;

        if extent.y > extent.x {
            axis = 1;
        }

        if extent.z > extent_slice[axis] {
            axis = 2;
        }

        let split_pos = aabb_min[axis] + extent_slice[axis] * 0.5;

        i = node.first_prim;
        j = node.first_prim + node.prim_count - 1;

        while i <= j {
            let triangle = triangles_indices[i].map(|vertex_id| vertices[vertex_id]);
            let centroid: [f32; 3] = get_centroid(triangle).into();

            if centroid[axis] < split_pos {
                i += 1;
            } else {
                triangles_indices.swap(i, j);
                j -= 1;
            }
        }
    }

    let left_count = i - bvh_data.nodes[node_id].first_prim;

    // abort split if one of the sides is empty
    if left_count == 0 || left_count == bvh_data.nodes[node_id].prim_count {
        return;
    }

    let new_left_node_id = bvh_data.node_count;
    bvh_data.node_count += 1;
    let new_right_node_id = bvh_data.node_count;
    bvh_data.node_count += 1;

    bvh_data.nodes[new_left_node_id].first_prim = bvh_data.nodes[node_id].first_prim;
    bvh_data.nodes[new_left_node_id].prim_count = left_count;

    bvh_data.nodes[new_right_node_id].first_prim = i;
    bvh_data.nodes[new_right_node_id].prim_count = bvh_data.nodes[node_id].prim_count - left_count;

    bvh_data.update_bounds(new_left_node_id, triangles_indices, vertices);
    bvh_data.update_bounds(new_right_node_id, triangles_indices, vertices);

    bvh_data.nodes[node_id].left_child = new_left_node_id as i32;
    bvh_data.nodes[node_id].right_child = new_right_node_id as i32;
    bvh_data.nodes[node_id].prim_count = 0;
    bvh_data.nodes[node_id].first_prim = 0;

    subdivide_bvh(bvh_data, new_left_node_id, triangles_indices, vertices);
    subdivide_bvh(bvh_data, new_right_node_id, triangles_indices, vertices);
}

pub fn build_bvh(
    triangles_indices: &mut Vec<[usize; 3]>,
    vertices: &Vec<cgmath::Vector3<f32>>,
) -> BvhData {
    let primitive_count = triangles_indices.len();
    let mut bvh_data = BvhData::new(primitive_count);

    bvh_data.root_id = 0;
    bvh_data.node_count = 1;
    bvh_data.nodes[bvh_data.root_id].first_prim = 0;
    bvh_data.nodes[bvh_data.root_id].prim_count = primitive_count;

    bvh_data.update_bounds(0, triangles_indices, vertices);
    subdivide_bvh(&mut bvh_data, 0, triangles_indices, vertices);

    return bvh_data;
}

pub trait DrawModel<'a> {
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
    );

    fn draw_model(&mut self, model: &'a Model, camera_bind_group: &'a wgpu::BindGroup);
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_mesh_instanced(mesh, material, 0..1, camera_bind_group);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, &material.bind_group, &[]);
        self.set_bind_group(1, camera_bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements as u32, 0, instances);
    }

    fn draw_model(&mut self, model: &'b Model, camera_bind_group: &'b wgpu::BindGroup) {
        self.draw_model_instanced(model, 0..1, camera_bind_group);
    }

    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            log::warn!("materials: {}", model.materials.len());
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(mesh, material, instances.clone(), camera_bind_group);
        }
    }
}
