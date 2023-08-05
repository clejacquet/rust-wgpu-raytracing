mod camera;
mod camera_control;
mod circle_camera_control;
mod model;
mod models;
mod resources;
mod texture;

use std::rc::Rc;
use std::{convert::TryInto, iter};

use cgmath::prelude::*;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use camera::Camera;
use camera_control::CameraController;
use circle_camera_control::CircleCameraController;
use model::ScreenVertex;
use models::sphere::Sphere;
use models::triangle_list::{TriangleData, TriangleList};
use texture::Texture;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

const VERTICES: &[ScreenVertex] = &[
    ScreenVertex {
        position: [-1.0, -1.0, 0.0],
        tex_coords: [0.0, 0.0],
    },
    ScreenVertex {
        position: [1.0, 1.0, 0.0],
        tex_coords: [1.0, 1.0],
    },
    ScreenVertex {
        position: [-1.0, 1.0, 0.0],
        tex_coords: [0.0, 1.0],
    },
    ScreenVertex {
        position: [-1.0, -1.0, 0.0],
        tex_coords: [0.0, 0.0],
    },
    ScreenVertex {
        position: [1.0, -1.0, 0.0],
        tex_coords: [1.0, 0.0],
    },
    ScreenVertex {
        position: [1.0, 1.0, 0.0],
        tex_coords: [1.0, 1.0],
    },
];

const NUM_INSTANCES_PER_ROW: u32 = 10;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = (OPENGL_TO_WGPU_MATRIX * camera.build_view_projection_matrix()).into();
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraInvUniform {
    viewmodel_inv: [[f32; 4]; 4],
    proj_inv: [[f32; 4]; 4],
    origin: [f32; 3],
    _padding: u32,
}

impl CameraInvUniform {
    fn new() -> Self {
        Self {
            viewmodel_inv: cgmath::Matrix4::identity().into(),
            proj_inv: cgmath::Matrix4::identity().into(),
            origin: cgmath::Vector3::zero().into(),
            _padding: 0,
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        // self.viewmodel_inv = (OPENGL_TO_WGPU_MATRIX * camera.build_view_inv_matrix()).into();
        self.viewmodel_inv = camera.build_view_inv_matrix().into();
        // self.proj_inv = camera.build_proj_inv_matrix().into();
        self.proj_inv = (OPENGL_TO_WGPU_MATRIX * camera.build_proj_inv_matrix()).into();
        self.origin = camera.eye.into();
    }
}

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from(self.rotation))
            .into(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    #[allow(dead_code)]
    model: [[f32; 4]; 4],
}

fn build_wasm_limits() -> wgpu::Limits {
    return wgpu::Limits {
        max_uniform_buffers_per_shader_stage: 11,
        max_storage_buffers_per_shader_stage: 0,
        max_storage_textures_per_shader_stage: 0,
        max_dynamic_storage_buffers_per_pipeline_layout: 0,
        max_storage_buffer_binding_size: 0,
        max_vertex_buffer_array_stride: 255,
        max_compute_workgroup_storage_size: 0,
        max_compute_invocations_per_workgroup: 0,
        max_compute_workgroup_size_x: 0,
        max_compute_workgroup_size_y: 0,
        max_compute_workgroup_size_z: 0,
        max_compute_workgroups_per_dimension: 0,

        // Most of the values should be the same as the downlevel defaults
        max_texture_dimension_1d: 4096,
        max_texture_dimension_2d: 4096,
        max_texture_dimension_3d: 256,
        max_texture_array_layers: 256,
        max_bind_groups: 4,
        max_bindings_per_bind_group: 640,
        max_dynamic_uniform_buffers_per_pipeline_layout: 8,
        max_sampled_textures_per_shader_stage: 16,
        max_samplers_per_shader_stage: 16,
        max_uniform_buffer_binding_size: 16 << 10,
        max_vertex_buffers: 8,
        max_vertex_attributes: 16,
        max_push_constant_size: 0,
        min_uniform_buffer_offset_alignment: 256,
        min_storage_buffer_offset_alignment: 256,
        max_inter_stage_shader_components: 60,
        max_buffer_size: 1 << 28,
    };
}

// impl InstanceRaw {
//     fn desc() -> wgpu::VertexBufferLayout<'static> {
//         use std::mem;
//         wgpu::VertexBufferLayout {
//             array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
//             // We need to switch from using a step mode of Vertex to Instance
//             // This means that our shaders will only change to use the next
//             // instance when the shader starts processing a new instance
//             step_mode: wgpu::VertexStepMode::Instance,
//             attributes: &[
//                 wgpu::VertexAttribute {
//                     offset: 0,
//                     // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
//                     // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
//                     shader_location: 5,
//                     format: wgpu::VertexFormat::Float32x4,
//                 },
//                 // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
//                 // for each vec4. We don't have to do this in code though.
//                 wgpu::VertexAttribute {
//                     offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
//                     shader_location: 6,
//                     format: wgpu::VertexFormat::Float32x4,
//                 },
//                 wgpu::VertexAttribute {
//                     offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
//                     shader_location: 7,
//                     format: wgpu::VertexFormat::Float32x4,
//                 },
//                 wgpu::VertexAttribute {
//                     offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
//                     shader_location: 8,
//                     format: wgpu::VertexFormat::Float32x4,
//                 },
//             ],
//         }
//     }
// }

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Screen {
    width: u32,
    height: u32,
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    // compute_pipeline: wgpu::ComputePipeline,
    // obj_model: Model,
    camera: Camera,
    camera_controller: Box<dyn CameraController>,
    camera_uniform: CameraUniform,
    camera_inv_uniform: CameraInvUniform,
    camera_buffer: wgpu::Buffer,
    camera_inv_buffer: wgpu::Buffer,
    screen_buffer: wgpu::Buffer,
    // camera_bind_group: wgpu::BindGroup,
    sphere: Sphere,
    sphere_front: Sphere,
    triangle_list: TriangleList,
    // compute_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_group: wgpu::BindGroup,
    compute_bind_group_front: wgpu::BindGroup,
    compute_bind_group_triangle: wgpu::BindGroup,
    compute_clear_buffer: wgpu::Buffer,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    screen_texture_bind_group: wgpu::BindGroup,
    // instances: Vec<Instance>,
    // instance_buffer: wgpu::Buffer,
    screen_vbo: wgpu::Buffer,
    screen_texture: Texture,
    depth_texture_input: Texture,
    depth_texture_output: Texture,
    window: Rc<Window>,
}

impl State {
    async fn new(window: Rc<Window>) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        log::warn!("WGPU setup");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(window.as_ref()) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        log::warn!("device and queue");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        build_wasm_limits()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                // Some(&std::path::Path::new("trace")), // Trace path
                None, // Trace path
            )
            .await
            .unwrap();

        log::warn!("Surface");
        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an Srgb surface texture. Using a different
        // one will result all the colors comming out darker. If you want to support non
        // Srgb surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let camera = Camera {
            eye: (0.0, 0.0, 0.0).into(),
            target: (0.0, 0.0, -1.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 60.0,
            znear: 0.1,
            zfar: 100.0,
        };
        let camera_controller = Box::new(CircleCameraController::new(0.2));

        let mut camera_inv_uniform = CameraInvUniform::new();
        camera_inv_uniform.update_view_proj(&camera);

        let camera_inv_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Inv Buffer"),
            contents: bytemuck::cast_slice(&[camera_inv_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let screen = Screen {
            width: config.width,
            height: config.height,
        };

        let screen_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Screen Buffer"),
            contents: bytemuck::cast_slice(&[screen]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let compute_clear_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compute Clear Buffer"),
            contents: vec![0; (32 * config.width * config.height) as usize]
                .into_boxed_slice()
                .as_ref(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
        });

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        const SPACE_BETWEEN: f32 = 3.0;
        let instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|z| {
                (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                    let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                    let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                    let position = cgmath::Vector3 { x, y: 0.0, z };

                    let rotation = if position.is_zero() {
                        cgmath::Quaternion::from_axis_angle(
                            cgmath::Vector3::unit_z(),
                            cgmath::Deg(0.0),
                        )
                    } else {
                        cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                    };

                    Instance { position, rotation }
                })
            })
            .collect::<Vec<_>>();

        // let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        // let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("Instance Buffer"),
        //     contents: bytemuck::cast_slice(&instance_data),
        //     usage: wgpu::BufferUsages::VERTEX,
        // });

        // let camera_bind_group_layout =
        //     device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        //         entries: &[wgpu::BindGroupLayoutEntry {
        //             binding: 0,
        //             visibility: wgpu::ShaderStages::VERTEX,
        //             ty: wgpu::BindingType::Buffer {
        //                 ty: wgpu::BufferBindingType::Uniform,
        //                 has_dynamic_offset: false,
        //                 min_binding_size: None,
        //             },
        //             count: None,
        //         }],
        //         label: Some("camera_bind_group_layout"),
        //     });

        // let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        //     layout: &camera_bind_group_layout,
        //     entries: &[wgpu::BindGroupEntry {
        //         binding: 0,
        //         resource: camera_buffer.as_entire_binding(),
        //     }],
        //     label: Some("camera_bind_group"),
        // });

        // log::warn!("Load model");
        // let obj_model =
        //     resources::load_model("cube.obj", &device, &queue, &texture_bind_group_layout)
        //         .await
        //         .unwrap();

        // let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        //     label: Some("shader.wgsl"),
        //     source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        // });

        let screen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("screenquad.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("screenquad.wgsl").into()),
        });

        let depth_texture_input = Texture::create_empty_texture(
            wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::R32Float,
            &device,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            "depth_texture_input",
        );

        let depth_texture_output = Texture::create_empty_texture(
            wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::R32Float,
            &device,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            "depth_texture_output",
        );

        let screen_vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let screen_texture = Texture::create_empty_texture(
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8Unorm,
            &device,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            "screen_texture",
        );

        let screen_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&screen_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&screen_texture.sampler),
                },
            ],
            label: Some("screen_texture_bind_group"),
        });

        let sphere = Sphere::new(&device, 0.4, cgmath::Vector3::new(0.6, 0.5, -4.0));

        let sphere_front = Sphere::new(&device, 0.4, cgmath::Vector3::new(0.4, 0.4, -3.0));

        let triangle_list = TriangleList::new(
            &device,
            vec![
                TriangleData::new(
                    cgmath::Vector3::new(0.4, 1.5, -4.0),
                    cgmath::Vector3::new(0.0, 1.0, -3.0),
                    cgmath::Vector3::new(0.8, 1.0, -3.0),
                ),
                TriangleData::new(
                    cgmath::Vector3::new(1.4, 1.5, -4.1),
                    cgmath::Vector3::new(1.0, 1.0, -3.0),
                    cgmath::Vector3::new(1.8, 1.0, -3.5),
                ),
            ],
        );

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: sphere.get_bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&screen_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_texture_input.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_texture_output.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: camera_inv_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: screen_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sphere.get_buffer().as_entire_binding(),
                },
            ],
            label: Some("compute_bind_group"),
        });

        let compute_bind_group_front = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: sphere_front.get_bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&screen_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_texture_input.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_texture_output.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: camera_inv_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: screen_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sphere_front.get_buffer().as_entire_binding(),
                },
            ],
            label: Some("compute_bind_group_front"),
        });

        let compute_bind_group_triangle = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: triangle_list.get_bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&screen_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_texture_input.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_texture_output.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: camera_inv_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: screen_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: triangle_list.get_buffer().as_entire_binding(),
                },
            ],
            label: Some("compute_bind_group_triangle"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                // bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &screen_shader,
                entry_point: "vs_main",
                // buffers: &[model::ModelVertex::desc(), InstanceRaw::desc()],
                buffers: &[ScreenVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &screen_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            // depth_stencil: Some(wgpu::DepthStencilState {
            //     format: texture::Texture::DEPTH_FORMAT,
            //     depth_write_enabled: true,
            //     depth_compare: wgpu::CompareFunction::Less,
            //     stencil: wgpu::StencilState::default(),
            //     bias: wgpu::DepthBiasState::default(),
            // }),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            // compute_pipeline,
            // obj_model,
            camera,
            camera_controller,
            camera_buffer,
            camera_inv_buffer,
            screen_buffer,
            // camera_bind_group,
            sphere,
            sphere_front,
            triangle_list,
            // compute_bind_group_layout,
            compute_bind_group,
            compute_bind_group_front,
            compute_bind_group_triangle,
            compute_clear_buffer,
            texture_bind_group_layout,
            screen_texture_bind_group,
            camera_uniform,
            camera_inv_uniform,
            // instances,
            // instance_buffer,
            depth_texture_input,
            depth_texture_output,
            screen_vbo,
            screen_texture,
            window,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.camera.aspect = self.config.width as f32 / self.config.height as f32;
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            self.depth_texture_input = Texture::create_empty_texture(
                wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureFormat::R32Float,
                &self.device,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                "depth_texture_input",
            );

            self.depth_texture_output = Texture::create_empty_texture(
                wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureFormat::R32Float,
                &self.device,
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                "depth_texture_output",
            );

            self.compute_clear_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Compute Clear Buffer"),
                        contents: vec![0; (32 * self.config.width * self.config.height) as usize]
                            .into_boxed_slice()
                            .as_ref(),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
                    });

            self.screen_texture = Texture::create_empty_texture(
                wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureFormat::Rgba8Unorm,
                &self.device,
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST,
                "screen_texture",
            );

            self.screen_texture_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.screen_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.screen_texture.sampler),
                        },
                    ],
                    label: Some("screen_texture_bind_group"),
                });

            self.compute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: self.sphere.get_bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.screen_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.depth_texture_input.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &self.depth_texture_output.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.camera_inv_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.screen_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.sphere.get_buffer().as_entire_binding(),
                    },
                ],
                label: Some("compute_bind_group"),
            });

            self.compute_bind_group_front =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: self.sphere_front.get_bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.screen_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.depth_texture_input.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &self.depth_texture_output.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.camera_inv_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.screen_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.sphere_front.get_buffer().as_entire_binding(),
                        },
                    ],
                    label: Some("compute_bind_group_front"),
                });

            self.compute_bind_group_triangle =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: self.triangle_list.get_bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.screen_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.depth_texture_input.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &self.depth_texture_output.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.camera_inv_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.screen_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.triangle_list.get_buffer().as_entire_binding(),
                        },
                    ],
                    label: Some("compute_bind_group_triangle"),
                });

            let screen = Screen {
                width: new_size.width,
                height: new_size.height,
            };

            self.queue
                .write_buffer(&self.screen_buffer, 0, bytemuck::cast_slice(&[screen]));

            self.camera_uniform.update_view_proj(&self.camera);
            self.camera_inv_uniform.update_view_proj(&self.camera);

            self.queue.write_buffer(
                &self.camera_buffer,
                0,
                bytemuck::cast_slice(&[self.camera_uniform]),
            );

            self.queue.write_buffer(
                &self.camera_inv_buffer,
                0,
                bytemuck::cast_slice(&[self.camera_inv_uniform]),
            );
        }
    }
    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.camera_inv_uniform.update_view_proj(&self.camera);

        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        self.queue.write_buffer(
            &self.camera_inv_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_inv_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let src_copy_buffer = wgpu::ImageCopyBuffer {
                buffer: &self.compute_clear_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(32 * self.config.width),
                    rows_per_image: Some(self.config.height),
                },
            };

            let dst_copy_texture = wgpu::ImageCopyTexture {
                texture: &self.screen_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            };

            encoder.copy_buffer_to_texture(
                src_copy_buffer,
                dst_copy_texture,
                wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        {
            let src_copy_buffer = wgpu::ImageCopyBuffer {
                buffer: &self.compute_clear_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(32 * self.config.width),
                    rows_per_image: Some(self.config.height),
                },
            };

            let dst_copy_texture = wgpu::ImageCopyTexture {
                texture: &self.depth_texture_input.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            };

            encoder.copy_buffer_to_texture(
                src_copy_buffer,
                dst_copy_texture,
                wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        {
            let src_copy_buffer = wgpu::ImageCopyBuffer {
                buffer: &self.compute_clear_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(32 * self.config.width),
                    rows_per_image: Some(self.config.height),
                },
            };

            let dst_copy_texture = wgpu::ImageCopyTexture {
                texture: &self.depth_texture_output.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            };

            encoder.copy_buffer_to_texture(
                src_copy_buffer,
                dst_copy_texture,
                wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });

            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.set_pipeline(self.sphere.get_pipeline());
            compute_pass.dispatch_workgroups(self.size.width, self.size.height, 1);
        }
        {
            let src_image_copy = wgpu::ImageCopyTexture {
                texture: &self.depth_texture_output.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            };

            let dst_image_copy = wgpu::ImageCopyTexture {
                texture: &self.depth_texture_input.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            };

            encoder.copy_texture_to_texture(
                src_image_copy,
                dst_image_copy,
                wgpu::Extent3d {
                    width: self.size.width,
                    height: self.size.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        {
            let mut compute_pass_front = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass Front"),
            });

            compute_pass_front.set_bind_group(0, &self.compute_bind_group_front, &[]);
            compute_pass_front.set_pipeline(self.sphere_front.get_pipeline());
            compute_pass_front.dispatch_workgroups(self.size.width, self.size.height, 1);
        }
        {
            let src_image_copy = wgpu::ImageCopyTexture {
                texture: &self.depth_texture_output.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            };

            let dst_image_copy = wgpu::ImageCopyTexture {
                texture: &self.depth_texture_input.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            };

            encoder.copy_texture_to_texture(
                src_image_copy,
                dst_image_copy,
                wgpu::Extent3d {
                    width: self.size.width,
                    height: self.size.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        {
            let mut compute_pass_triangle =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass Triangle"),
                });

            compute_pass_triangle.set_bind_group(0, &self.compute_bind_group_triangle, &[]);
            compute_pass_triangle.set_pipeline(self.triangle_list.get_pipeline());
            compute_pass_triangle.dispatch_workgroups(self.size.width, self.size.height, 1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                // depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                //     view: &self.depth_texture.view,
                //     depth_ops: Some(wgpu::Operations {
                //         load: wgpu::LoadOp::Clear(1.0),
                //         store: true,
                //     }),
                //     stencil_ops: None,
                // }),
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.screen_vbo.slice(..));
            render_pass.set_bind_group(0, &self.screen_texture_bind_group, &[]);
            render_pass.draw(0..6, 0..1);

            // render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            // render_pass.draw_model_instanced(
            //     &self.obj_model,
            //     0..self.instances.len() as u32,
            //     &self.camera_bind_group,
            // );
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Could't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new();
    let title = env!("CARGO_PKG_NAME");
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        // .with_inner_size(winit::dpi::LogicalSize::new(600, 600))
        .build(&event_loop)
        .unwrap();

    let window_rc = Rc::new(window);

    #[cfg(target_arch = "wasm32")]
    {
        let get_full_size = || {
            // TODO Not sure how to get scrollbar dims
            let scrollbars = 0.0;
            let win = web_sys::window().unwrap();
            // `inner_width` corresponds to the browser's `self.innerWidth` function, which are in
            // Logical, not Physical, pixels
            winit::dpi::LogicalSize::new(
                win.inner_width().unwrap().as_f64().unwrap() - scrollbars,
                win.inner_height().unwrap().as_f64().unwrap() - scrollbars,
            )
        };

        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        window_rc.set_inner_size(get_full_size());

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window_rc.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");

        // resize of our winit::Window whenever the browser window changes size.
        {
            let window_closure_ref = window_rc.clone();
            let closure = wasm_bindgen::closure::Closure::wrap(Box::new(move |e: web_sys::Event| {
                let size = get_full_size();

                window_closure_ref.set_inner_size(size)
            }) as Box<dyn FnMut(_)>);

            let win = web_sys::window().unwrap();

            win.add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref())
                .unwrap();
            closure.forget();
        }
    }

    // State::new uses async code, so we're going to wait for it to finish
    let mut state = State::new(window_rc.clone()).await;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared => state.window().request_redraw(),
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state.resize(state.size)
                    }
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // We're ignoring timeouts
                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                }
            }
            _ => {}
        }
    });
}
