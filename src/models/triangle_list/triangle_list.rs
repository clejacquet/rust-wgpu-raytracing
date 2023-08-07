use wgpu::util::BufferInitDescriptor;
use wgpu::util::DeviceExt;

use crate::model;

pub struct TriangleList {
    model: model::Model,
    bvh: model::BvhData,
    material_buffer: wgpu::Buffer,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TriangleBufferData {
    p0: [f32; 3],
    pad0: f32,
    p1: [f32; 3],
    pad1: f32,
    p2: [f32; 3],
    pad2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialData {
    ambient: [f32; 3],
    pad0: f32,
    diffuse: [f32; 3],
    pad1: f32,
    specular: [f32; 3],
    pad2: f32,
}

impl TriangleBufferData {
    fn new(p0: cgmath::Vector3<f32>, p1: cgmath::Vector3<f32>, p2: cgmath::Vector3<f32>) -> Self {
        return Self {
            p0: p0.into(),
            pad0: 0.0,
            p1: p1.into(),
            pad1: 0.0,
            p2: p2.into(),
            pad2: 0.0,
        };
    }
}

impl MaterialData {
    fn new(ambient: cgmath::Vector3<f32>, diffuse: cgmath::Vector3<f32>, specular: cgmath::Vector3<f32>) -> Self {
        return Self {
            ambient: ambient.into(),
            pad0: 0.0,
            diffuse: diffuse.into(),
            pad1: 0.0,
            specular: specular.into(),
            pad2: 0.0,
        };
    }
}

#[derive(Clone)]
pub struct TriangleData {
    p0: cgmath::Vector3<f32>,
    p1: cgmath::Vector3<f32>,
    p2: cgmath::Vector3<f32>,
}

impl TriangleData {
    pub fn new(
        p0: cgmath::Vector3<f32>,
        p1: cgmath::Vector3<f32>,
        p2: cgmath::Vector3<f32>,
    ) -> Self {
        Self { p0, p1, p2 }
    }
}

impl TriangleList {
    pub fn new(device: &wgpu::Device, model: model::Model) -> Self {
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("compute_bind_group_layout"),
            });

        let compute_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("compute_texture_bind_group_layout"),
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute_pipeline_layout"),
                bind_group_layouts: &[
                    &compute_bind_group_layout,
                    &compute_texture_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: Some(&compute_pipeline_layout),
            entry_point: "main",
            module: &compute_shader,
        });

        let material_data = MaterialData::new(model.materials[0].ambient, model.materials[0].diffuse, model.materials[0].specular);

        let material_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("triangle_material_buffer"),
            contents: bytemuck::cast_slice(&[material_data]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bvh = model.build_bvh(&mut model.meshes[0].triangle_indices.clone(), &model.meshes[0].vertices);

        return Self {
            model,
            bvh,
            material_buffer,
            compute_bind_group_layout,
            compute_pipeline,
        };
    }

    pub fn get_vertex_buffer(&self) -> &wgpu::Buffer {
        &self.model.meshes[0].vertex_buffer
    }

    pub fn get_index_buffer(&self) -> &wgpu::Buffer {
        &self.model.meshes[0].index_buffer
    }

    pub fn get_material_buffer(&self) -> &wgpu::Buffer {
        &self.material_buffer
    }

    pub fn get_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.compute_bind_group_layout
    }

    pub fn get_texture_bind_group(&self) -> &wgpu::BindGroup {
        &self.model.materials[0].bind_group
    }

    pub fn get_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.compute_pipeline
    }
}
