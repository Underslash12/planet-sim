// lib.rs

use env_logger::init;
use log::{error, info};
use core::f32;
use std::{collections::VecDeque, f32::consts::PI};
use web_time::{Duration, Instant};
use std::thread;
use std::sync::{Arc, Mutex};
use std::any::type_name;

use winit::{
    dpi::PhysicalSize, event::*, event_loop::EventLoop, keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowBuilder}
};
use wgpu::{util::DeviceExt, vertex_attr_array};
use cgmath::{perspective, prelude::*, Point3, Rad, Deg, Vector3, Vector4, Matrix4, Quaternion};

#[cfg(target_arch="wasm32")]    
use wasm_bindgen::{prelude::*, JsCast, closure};
#[cfg(target_arch="wasm32")]
use web_sys::{HtmlButtonElement, HtmlInputElement, HtmlElement, Document};

mod texture;
mod model;
use model::Vertex;
mod resources;
mod sim;
use sim::{AstroBody, AstroBodyInstanceRaw, PlanetSim};


// #[repr(C)]
// #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
// struct Vertex {
//     position: [f32; 3],
//     tex_coords: [f32; 2],
// }

// impl Vertex {
//     // two attributes, the first being the position, and the second the texture coordinates
//     // attributes are what is referenced by @location in the wgsl shader code, and are shared between
//     // different invocations (not sure if also shared over different shader types)
//     const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
//         vertex_attr_array![0 => Float32x3, 1 => Float32x2];

//     // get the vertexbufferlayout associated with the Vertex struct
//     // the vertexbufferlayout is just how each vertex is laid out within the buffer
//     fn desc() -> wgpu::VertexBufferLayout<'static> {
//         wgpu::VertexBufferLayout {
//             array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
//             step_mode: wgpu::VertexStepMode::Vertex,
//             // the attributes specify how each vertex should be further divided
//             attributes: &Self::ATTRIBUTES,
//         }
//     }
// }

// // a simple pentagon
// const VERTICES: &[Vertex] = &[
//     Vertex { position: [-0.0868241, 0.49240386, 0.0], tex_coords: [0.4131759, 1.0 - 0.99240386], }, // A
//     Vertex { position: [-0.49513406, 0.06958647, 0.0], tex_coords: [0.0048659444, 1.0 - 0.56958647], }, // B
//     Vertex { position: [-0.21918549, -0.44939706, 0.0], tex_coords: [0.28081453, 1.0 - 0.05060294], }, // C
//     Vertex { position: [0.35966998, -0.3473291, 0.0], tex_coords: [0.85967, 1.0 - 0.1526709], }, // D
//     Vertex { position: [0.44147372, 0.2347359, 0.0], tex_coords: [0.9414737, 1.0 - 0.7347359], }, // E
// ];

// // vertex buffer indices
// const INDICES: &[u16] = &[
//     0, 1, 4,
//     1, 2, 4,
//     2, 3, 4,
// ];

// a cube
// const VERTICES: &[Vertex] = &[
//     Vertex { position: [-1.0, -1.0, -1.0], tex_coords: [-1.0, -1.0], }, 
//     Vertex { position: [1.0, -1.0, -1.0], tex_coords: [1.0, -1.0], }, 
//     Vertex { position: [-1.0, -1.0, 1.0], tex_coords: [-1.0, 1.0], }, 
//     Vertex { position: [1.0, -1.0, 1.0], tex_coords: [1.0, 1.0], }, 
//     Vertex { position: [-1.0, 1.0, -1.0], tex_coords: [-1.0, -1.0], }, 
//     Vertex { position: [1.0, 1.0, -1.0], tex_coords: [1.0, -1.0], }, 
//     Vertex { position: [-1.0, 1.0, 1.0], tex_coords: [-1.0, 1.0], }, 
//     Vertex { position: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0], }, 
// ];

// const INDICES: &[u16] = &[
//     // bottom
//     0, 1, 3,
//     0, 3, 2,
//     // front
//     0, 5, 1,
//     0, 4, 5,
//     // left
//     0, 2, 4,
//     2, 6, 4,
//     // back
//     2, 7, 6,
//     2, 3, 7,
//     // right
//     1, 7, 3,
//     1, 5, 7,
//     // top
//     4, 7, 5,
//     4, 6, 7,
// ];


// now camera stuff
// camera matrix for converting between camera types
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

// perspective camera struct
struct Camera {
    pos: Vector3<f32>,
    // specify the camera rotation as raw yaw and pitch since we don't want it to roll
    // and the yaw can be easily restricted here
    yaw: f32,
    pitch: f32,
    up: Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self, offset: Vector3<f32>) -> Matrix4<f32> {
        // let view = Matrix4::look_at_rh(self.eye, self.target, self.up);
        
        // want to convert from world space to cam space, so compute the inverse of what we normally would
        let translation_inv = Matrix4::from_translation(-self.pos + (-offset));
        let yaw_inv = Matrix4::from_angle_y(Rad(self.yaw)).transpose();
        let pitch_inv = Matrix4::from_angle_x(Rad(self.pitch)).transpose();
        // let initial_rotation: Matrix4<f32> = Matrix4::look_at_rh(Point3::from_vec(Vector3::zero()), Point3::from_vec(Vector3::unit_z()), self.up);
        
        // so instead of applying pitch -> yaw -> translation, we apply translation^-1 -> yaw^-1 -> pitch^-1
        let view = pitch_inv * yaw_inv * translation_inv;
        let proj = perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * proj * view
    }

    fn view(&self) -> Matrix4<f32> {
        let yaw_rot = Matrix4::from_angle_y(Rad(-self.yaw));
        let pitch_rot = Matrix4::from_angle_x(Rad(-self.pitch));
        yaw_rot * pitch_rot
    }
}

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // We can't use cgmath with bytemuck directly, so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        use SquareMatrix;
        Self {
            view_proj: Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera, offset: Vector3<f32>) {
        self.view_proj = camera.build_view_projection_matrix(offset).into();
    }
}

// basic camera controller
struct CameraInput {
    // each input is how much it is moving/turning in that direction
    move_forward_pressed: bool,
    move_backward_pressed: bool,
    move_right_pressed: bool,
    move_left_pressed: bool,
    move_up_pressed: bool,
    move_down_pressed: bool,
    turn_right_pressed: bool,
    turn_left_pressed: bool,
    turn_up_pressed: bool,
    turn_down_pressed: bool,
}

impl CameraInput {
    fn new() -> Self {
        Self {
            move_forward_pressed: false,
            move_backward_pressed: false,
            move_right_pressed: false,
            move_left_pressed: false,
            move_up_pressed: false,
            move_down_pressed: false,
            turn_right_pressed: false,
            turn_left_pressed: false,
            turn_up_pressed: false,
            turn_down_pressed: false,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    KeyCode::KeyW => {
                        self.move_forward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyS => {
                        self.move_backward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyD => {
                        self.move_right_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyA => {
                        self.move_left_pressed = is_pressed;
                        true
                    }
                    KeyCode::Space => {
                        self.move_up_pressed = is_pressed;
                        true
                    }
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                        self.move_down_pressed = is_pressed;
                        true
                    }
                    KeyCode::ArrowRight => {
                        self.turn_right_pressed = is_pressed;
                        true
                    }
                    KeyCode::ArrowLeft => {
                        self.turn_left_pressed = is_pressed;
                        true
                    }
                    KeyCode::ArrowUp => {
                        self.turn_up_pressed = is_pressed;
                        true
                    }
                    KeyCode::ArrowDown => {
                        self.turn_down_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            // WindowEvent::CursorMoved { device_id, position } => {
            //     println!("{:?}", position);
            //     true
            // },
            _ => false,
        }
    }

    fn move_forward_input(&self) -> f32 {
        let mut result = 0.0;
        // these are switched since +z is out of the screen, not into it
        if self.move_forward_pressed {
            result -= 1.0;
        } 
        if self.move_backward_pressed {
            result += 1.0;
        }
        result
    }

    fn move_right_input(&self) -> f32 {
        let mut result = 0.0;
        if self.move_right_pressed {
            result += 1.0;
        } 
        if self.move_left_pressed {
            result -= 1.0;
        }
        result
    }

    fn move_up_input(&self) -> f32 {
        let mut result = 0.0;
        if self.move_up_pressed {
            result += 1.0;
        } 
        if self.move_down_pressed {
            result -= 1.0;
        }
        result
    }

    fn turn_right_input(&self) -> f32 {
        let mut result = 0.0;
        // for yaw, i think this is just arbitrary (might be wrong), but will flip this as well
        if self.turn_right_pressed {
            result -= 1.0;
        } 
        if self.turn_left_pressed {
            result += 1.0;
        }
        result
    }

    fn turn_up_input(&self) -> f32 {
        let mut result = 0.0;
        if self.turn_up_pressed {
            result += 1.0;
        } 
        if self.turn_down_pressed {
            result -= 1.0;
        }
        result
    }
}

struct CameraController {
    input: CameraInput, 
    speed: f32,
}

impl CameraController {
    fn new(speed: f32) -> CameraController {
        CameraController {
            input: CameraInput::new(),
            speed,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        self.input.process_events(event)
    }

    fn update_camera(&self, camera: &mut Camera) {
        // update the position
        // have to perform a change of basis to get to camera space
        let translation_vector = self.speed * Vector4::new(
            self.input.move_right_input(), 
            0.0,
            self.input.move_forward_input(),
            1.0,
        );

        // i personally want the up and down not to depend on camera orientation, but maybe ill make that a toggle
        let yaw = Matrix4::from_angle_y(Rad(camera.yaw));
        camera.pos += (yaw * translation_vector).xyz();
        camera.pos.y += self.input.move_up_input() * self.speed;

        // update the rotation
        camera.yaw += self.input.turn_right_input() * self.speed;
        camera.yaw %= 2.0 * f32::consts::PI;
        camera.pitch += self.input.turn_up_input() * self.speed;
        camera.pitch = camera.pitch.clamp(-f32::consts::PI / 2.0, f32::consts::PI / 2.0);
    }
}


// counts frames since program start (roughly)
struct FrameCounter {
    frame: usize,
    frame_times: VecDeque<Instant>,
}

impl FrameCounter {
    fn new() -> FrameCounter {
        FrameCounter {
            frame: 0,
            frame_times: VecDeque::new(),
        }
    }

    // should be called every update cycle
    fn update(&mut self) {
        self.frame += 1;
        self.frame_times.push_back(Instant::now());

        // only keep frames within the past second
        while self.frame_times.len() > 0 {
            let time_diff: Duration = *self.frame_times.back().unwrap() - *self.frame_times.front().unwrap();
            if time_diff > Duration::from_secs(1) {
                self.frame_times.pop_front();
            } else {
                break;
            }
        }
    }

    fn fps(&self) -> f64 {
        if self.frame_times.len() > 0 {
            return self.frame_times.len() as f64;
        }
        0.0
    }

    // computes the time between the previous two frames
    fn delta_time(&self) -> Duration {
        if self.frame_times.len() >= 2 {
            let frames = self.frame_times.len();
            return *self.frame_times.get(frames - 1).unwrap() - *self.frame_times.get(frames - 2).unwrap();
        }
        Duration::from_secs(0)
    }   

    fn clamped_delta_time(&self) -> Duration {
        let delta_time = self.delta_time();
        let max_delta_time = Duration::from_secs_f64(0.1);
        if delta_time > max_delta_time {
            max_delta_time
        } else {
            delta_time
        }
    }
}


// webgpu state including the surface, device, and render pipeline
struct State<'a> {
    frame_counter: FrameCounter,
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: &'a Window,
    render_pipeline: wgpu::RenderPipeline,
    // num_vertices: u32,
    // vertex_buffer: wgpu::Buffer,
    // num_indices: u32,
    // index_buffer: wgpu::Buffer, 
    // obj_model: model::Model,
    sphere_mesh: model::Mesh,
    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
    depth_texture: texture::Texture,
    // texture_array: texture::Texture,
    // texture_array_bind_group: wgpu::BindGroup,
    texture_array_material: model::Material,
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    // instances: Vec<Instance>,
    planet_sim: Arc<Mutex<PlanetSim>>,
    sec_per_sec: Arc<Mutex<u32>>,
    instance_buffer: wgpu::Buffer,
}

impl<'a> State<'a> {
    // Creating some of the wgpu types requires async code
    pub async fn new(window: &'a Window, planet_sim: PlanetSim) -> State<'a> {
        // window.inner_size() returns 0 on the web (probably the way the html is written), so for now, just hardcode it for the wasm
        #[cfg(target_arch = "wasm32")]
        let mut size = PhysicalSize::new(640, 640);
        #[cfg(not(target_arch = "wasm32"))]
        let size = window.inner_size();
        error!("Window Size: {:?}", window.inner_size());
        
        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch="wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch="wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        // create the surface that will be rendered to
        let surface = instance.create_surface(window).unwrap();

        // since I am using wasm, we need to use request_adapter, which gives us a GPU adapter 
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        // using the adapater, we can get the underlying device and command queue
        // since we are targeting wasm, the limits are set by webgl
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web, we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
                memory_hints: Default::default(),
            },
            None, // Trace path
        ).await.unwrap();

        // configure the surface
        // using AutoVsync for now, but might change it later depending on performance goals
        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        
        // load the vertex and fragment shaders from the same file
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        // alternatively, could use the following
        // let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));        

        // texture stuff
        let diffuse_bytes = include_bytes!("../happy_tree.png");
        let diffuse_texture = texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy_tree.png").unwrap(); 

        // define the layout of the bind groups we will be using, in this case, binding 0 is the sampled texture
        // and binding 1 is the sampler
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
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });
            
        // create the bind group using the above layout
        let diffuse_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    }
                ],
                label: Some("diffuse_bind_group"),
            }
        );

        // let test_texture_array = texture::Texture::from_bytes_array(&device, &queue, bytes, "planet_texture_array");
        let texture_array = resources::load_texture_array(&["textures/2k_mercury.jpg", "textures/2k_venus_atmosphere.jpg"], &device, &queue, "planet_texture_array").await.unwrap();
        
        let texture_array_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_array_bind_group_layout"),
            });
            
        // create the bind group using the above layout
        let texture_array_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_array_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&texture_array.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&texture_array.sampler),
                    }
                ],
                label: Some("texture_array_bind_group"),
            }
        );
        
        // create the camera
        let camera = Camera {
            // position the camera 1 unit up and 2 units back
            // +z is out of the screen
            // eye: (0.0, 1.0, 2.0).into(),
            // have it look at the origin
            // target: (0.0, 0.0, 0.0).into(),
            // which way is "up"
            pos: Vector3::zero(),
            yaw: 0.0,
            pitch: 0.0,
            up: Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let camera_controller = CameraController::new(0.03);

        // create the camera uniform and buffer
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, planet_sim.get_focused().unwrap().get_low_precision_position());

        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        // now we need to create the bind group for the camera, which requires another bind layout
        // this one just has a single binding, namely a buffer
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("camera_bind_group_layout"),
        });
        
        // camera uniform bind group
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("camera_bind_group"),
        });
        
        // create the instance buffer for the astronomical bodies
        let instance_data = planet_sim.instance_data();
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("AstroBody Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        // depth texture stuff
        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");
        
        // load the sphere mesh
        let sphere_mesh = resources::load_mesh("sphere_mesh.obj", &device, &queue, &texture_bind_group_layout)
            .await.unwrap();

        // the layout for the pipeline, useful for hotswapping pipelines i think
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &texture_array_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        // create the render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    model::ModelVertex::desc(),
                    AstroBodyInstanceRaw::desc(),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // cull_mode: None,
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // 1.
                stencil: wgpu::StencilState::default(), // 2.
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None, 
        });

        // // want to store the number of vertices for rendering
        // let num_vertices = VERTICES.len() as u32;
        // // create a vertex buffer for, you guessed it, the vertices
        // let vertex_buffer = device.create_buffer_init(
        //     &wgpu::util::BufferInitDescriptor {
        //         label: Some("Vertex Buffer"),
        //         contents: bytemuck::cast_slice(VERTICES),
        //         usage: wgpu::BufferUsages::VERTEX,
        //     }
        // );

        // // number of indices
        // let num_indices = INDICES.len() as u32;
        // // index buffer to reuse the vertices
        // // though an index buffer still has some waste, it wastes 2 bytes per vertex instead of sizeof(vec3f) = 12 bytes per vertex
        // let index_buffer = device.create_buffer_init(
        //     &wgpu::util::BufferInitDescriptor {
        //         label: Some("Index Buffer"),
        //         contents: bytemuck::cast_slice(INDICES),
        //         usage: wgpu::BufferUsages::INDEX,
        //     }
        // );
            
        let texture_array_material = model::Material {
            name: String::from("PlanetTextureMaterial"),
            diffuse_texture: texture_array,
            bind_group: texture_array_bind_group,
        };

        // create the state
        State {
            frame_counter: FrameCounter::new(),
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            // num_vertices,
            // vertex_buffer,
            // num_indices,
            // index_buffer,
            // obj_model,
            sphere_mesh,
            diffuse_bind_group,
            diffuse_texture,
            depth_texture,
            // texture_array,
            // texture_array_bind_group,
            texture_array_material,
            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            // instances,
            instance_buffer,
            planet_sim: Arc::new(Mutex::new(planet_sim)), 
            sec_per_sec: Arc::new(Mutex::new(1)),
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    // reconfigure the surface when resizing the window
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;

            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");

            self.surface.configure(&self.device, &self.config);
        }
    }

    // for handling inputs, currently unused
    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    // update the state
    fn update(&mut self) {
        self.frame_counter.update();

        self.camera_controller.update_camera(&mut self.camera);
        self.camera.aspect = self.config.width as f32 / self.config.height as f32;
        self.camera_uniform.update_view_proj(&self.camera, self.planet_sim.lock().unwrap().get_focused().unwrap().get_low_precision_position());
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));

        self.planet_sim.lock().unwrap().update(*self.sec_per_sec.lock().unwrap() * self.frame_counter.clamped_delta_time());
        let instance_data = self.planet_sim.lock().unwrap().instance_data();
        self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instance_data));
    }

    // render whatever needs to be rendered onto the surface
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // get the texture to be rendered to
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // create the command encoder (to encode commands to be sent to the gpu)
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        // block drops the encoder so that we can call finish on it after
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(
                                wgpu::Color {
                                    r: 0.1,
                                    g: 0.2,
                                    b: 0.3,
                                    a: 1.0,
                                }
                            ),
                            store: wgpu::StoreOp::Store,
                        }
                    })
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline); 
            // add in a bind group for our texture
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            // // set the bind group for our camera binding
            // render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            // // need to actually assign the buffer we created to the renderer (since it needs a specific slot)
            // render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            // set the instance buffer to be slot 1
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            // // as with the vertex buffer, we have to set the active index buffer, but this time there is only one slot (hence active)
            // render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16); 
            // instead of using draw, since we are using an index buffer, we have to use draw_indexed
            // render_pass.draw_indexed(0..self.num_indices, 0, 0..1);      
            // render_pass.draw_indexed(0..self.num_indices, 0, 0..self.planet_sim.lock().unwrap().len() as u32);  
            
            use model::DrawModel;
            let instances = self.planet_sim.lock().unwrap().len() as u32;
            render_pass.draw_mesh_instanced(&self.sphere_mesh, &self.texture_array_material, 0..instances, &self.camera_bind_group);
            // render_pass.draw_mesh_instanced(mesh, material, 0..self.planet_sim.lock().unwrap().len() as u32, &self.camera_bind_group);

            // render_pass.draw_model_instanced(
            //     &self.obj_model,
            //     0..self.planet_sim.lock().unwrap().len() as u32,
            //     &self.camera_bind_group,
            // );
        }
        
        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    
        Ok(())
    }
}


// get a generic html element by id using web_sys
#[cfg(target_arch = "wasm32")]
fn get_html_element_by_id<T: JsCast + Clone>(doc: &Document, id: &str) -> T {
    let element: web_sys::Element = doc.get_element_by_id(id).expect(&format!("Unable to get element {}", id));
    element.dyn_ref::<T>().expect(&format!("Unable to convert {} of type web_sys::Element into type {}", id, type_name::<T>())).clone()
}

// given an element and an htmlelement event, register some callback with it
// this is complicated, and if it breaks, go back to "text input working" on branch js_bindings for a working but more repetitive method 
#[cfg(target_arch = "wasm32")]
fn register_js_callback<'a, E, F, C>(element: &'a E, event_function: F, callback: C) 
    where 
        E: AsRef<HtmlElement>,
        F: Fn(&'a HtmlElement, Option<&web_sys::js_sys::Function>),
        C: Fn() + 'static,
{
    let closure: Closure<dyn FnMut()> = Closure::wrap(Box::new(callback) as Box<dyn FnMut()>);
    event_function(element.as_ref(), Some(closure.as_ref().unchecked_ref()));
    closure.forget();
}

// given an element and an htmlelement event, register some callback with it where the callback can also access a web_sys::event parameter
#[cfg(target_arch = "wasm32")]
fn register_js_callback_with_event<'a, E, F, C>(element: &'a E, event_function: F, callback: C) 
    where 
        E: AsRef<HtmlElement>,
        F: Fn(&'a HtmlElement, Option<&web_sys::js_sys::Function>),
        C: Fn(web_sys::Event) + 'static,
{
    let closure: Closure<dyn FnMut(_)> = Closure::wrap(Box::new(callback) as Box<dyn FnMut(_)>);
    event_function(element.as_ref(), Some(closure.as_ref().unchecked_ref()));
    closure.forget();
}


#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run() {
    // log logs and panics in the js console if targeting wasm
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }
    
    let event_loop = EventLoop::new().unwrap();
    let window: Window = WindowBuilder::new().build(&event_loop).unwrap();
    // window.set_cursor_visible(false);
    // #[cfg(target_arch = "wasm32")]
    // window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
    // #[cfg(not(target_arch = "wasm32"))]
    // window.set_cursor_grab(winit::window::CursorGrabMode::Confined);

    // if targetting wasm, add the canvas that is displaying the webgl to the document
    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        let _ = window.request_inner_size(PhysicalSize::new(640, 640));
        
        use winit::platform::web::WindowExtWebSys;
        use web_sys::{Element, HtmlButtonElement, Node};

        // get the html document
        let document = web_sys::window().expect("Unable to get window").document().expect("Unable to get document");
        
        // construct the canvas from the winit window
        let body = document.get_element_by_id("planet-sim").expect("Unable to get document body");
        let canvas = Element::from(window.canvas().expect("Unable to create winit canvas"));
        // want the canvas to be on the left (in front of the )
        let first_child = body.child_nodes().item(0).expect("Couldn't get child 0 of body");
        body.insert_before(&canvas, Some(&first_child)).expect("Failed to insert canvas into document body");
    } 

    // test planet sim
    let mut planet_sim = PlanetSim::new(5.0);
    planet_sim.add(AstroBody::new(
        "Test 1",
        0,
        100.0, 
        0.5, 
        Vector3::new(0.0, 0.0, 0.0),
        // Vector3::new(0.0, 0.0, 1.0),
        Vector3::zero(),
        Quaternion::zero(),
        Vector3::unit_z(),
        0.0,
    ));
    planet_sim.add(AstroBody::new(
        "Test 2",
        1,
        1.0, 
        0.5, 
        Vector3::new(-5.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, -10.0),
        Quaternion::zero(),
        Vector3::unit_z(),
        0.0,
    ));
    // planet_sim.add(AstroBody::new(
    //     "Test 3",
    //     1.0, 
    //     0.5, 
    //     Vector3::new(-1.0, 0.5, 0.5),
    //     Vector3::new(0.0, 0.0, -1.0),
    //     Quaternion::zero(),
    //     Vector3::unit_z(),
    //     0.0,
    // ));
    planet_sim.set_focused(Some("Test 1"));

    // create the state
    let mut state = State::new(&window, planet_sim).await;
    let mut surface_configured = false;

    // add events to the html elements
    #[cfg(target_arch = "wasm32")]
    {
        // get the html document
        let document = web_sys::window().expect("Unable to get window").document().expect("Unable to get document");

        // test button
        let button = get_html_element_by_id::<HtmlButtonElement>(&document, "test-button");
        let planet_sim = state.planet_sim.clone();
        let button_onclick = move || {
            error!("Button Clicked!! {:?}", &planet_sim.lock().unwrap().get_focused().unwrap().label);
        };
        register_js_callback(&button, HtmlElement::set_onclick, button_onclick);

        // timescale adjuster
        let timescale_input = get_html_element_by_id::<HtmlInputElement>(&document, "dt-input");
        let sec_per_sec = state.sec_per_sec.clone();
        let timescale_onchange = move |event: web_sys::Event| {
            let target = event.target().unwrap();
            let input_element = target.dyn_into::<HtmlInputElement>().unwrap();
            let value = input_element.value();
            if let Ok(dt) = value.parse::<u32>() {
                *sec_per_sec.lock().unwrap() = dt;
                error!("New timescale: {}sec/sec", dt);
            } else {
                error!("{} is not a valid timescale", value);
            }
        };
        register_js_callback_with_event(&timescale_input, HtmlElement::set_onchange, timescale_onchange);
    } 

    event_loop.run(move |event, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => {
                if !state.input(event) { 
                    match event {
                        WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    state: ElementState::Pressed,
                                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                                    ..
                                },
                            ..
                        } => control_flow.exit(),
                        WindowEvent::Resized(physical_size) => {
                            log::info!("physical_size: {physical_size:?}");
                            surface_configured = true;
                            state.resize(*physical_size);
                        }
                        WindowEvent::RedrawRequested => {
                            // This tells winit that we want another frame after this one
                            state.window().request_redraw();

                            // just kidding, do need to check if the surface is configured
                            if !surface_configured {
                                return;
                            }
                
                            state.update();
                            match state.render() {
                                Ok(_) => {},
                                // Reconfigure the surface if it's lost or outdated
                                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                    state.resize(state.size)
                                },
                                // The system is out of memory, we should probably quit
                                Err(wgpu::SurfaceError::OutOfMemory | wgpu::SurfaceError::Other) => {
                                    log::error!("OutOfMemory");
                                    control_flow.exit();
                                },
                                // This happens when the a frame takes too long to present
                                Err(wgpu::SurfaceError::Timeout) => {
                                    log::warn!("Surface timeout")
                                },
                            }
                        }
                        _ => {}
                    }
                }
            }  
            _ => {}
        }
    }).expect("Failed to run the event loop");
}

