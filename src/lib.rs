// lib.rs

use env_logger::init;
use log::{error, info};
use winit::dpi::PhysicalPosition;
use core::f32;
use std::{collections::VecDeque, f32::consts::PI};
use web_time::{Duration, Instant};
use std::thread;
use std::sync::{Arc, Mutex};
use std::any::type_name;

use winit::{
    dpi::PhysicalSize, event::*, event_loop::EventLoop, keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowBuilder}
};
use wgpu::{vertex_attr_array, VertexBufferLayout, Device, SurfaceConfiguration, BindGroupLayout, BindGroup, Buffer, VertexStepMode, BufferAddress, util::DeviceExt};
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


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PointVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl PointVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        vertex_attr_array![0 => Float32x3, 1 => Float32x4];
}

impl Vertex for PointVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        VertexBufferLayout {
            array_stride: mem::size_of::<PointVertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

// get a pointvertex with position and color corresponding to some astronomical body
fn point_vertex_from_astro_body(obj: &AstroBody, render_scale: f64) -> PointVertex {
    let pos = obj.get_low_precision_position(render_scale);
    PointVertex {
        position: pos.into(),
        // color: obj.color,
        color: [0.0, 0.0, 1.0, 1.0],
    }
}


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
    // no offset needed since the camera is (in this case) always going to "targeting" something around 0
    fn build_view_projection_matrix(&self) -> Matrix4<f32> {
        // let view = Matrix4::look_at_rh(self.eye, self.target, self.up);
        
        // want to convert from world space to cam space, so compute the inverse of what we normally would
        let translation_inv = Matrix4::from_translation(-self.pos);
        let yaw_inv = Matrix4::from_angle_y(Rad(self.yaw)).transpose();
        let pitch_inv = Matrix4::from_angle_x(Rad(self.pitch)).transpose();
        // let initial_rotation: Matrix4<f32> = Matrix4::look_at_rh(Point3::from_vec(Vector3::zero()), Point3::from_vec(Vector3::unit_z()), self.up);
        
        // so instead of applying pitch -> yaw -> translation, we apply translation^-1 -> yaw^-1 -> pitch^-1
        let view = pitch_inv * yaw_inv * translation_inv;
        let proj = perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * proj * view
    }

    fn get_high_precision_position(&self) -> Vector3<f64> {
        Vector3::new(self.pos.x as f64, self.pos.y as f64, self.pos.z as f64)
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

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
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
    scroll_delta: f32,
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
            scroll_delta: 0.0,
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
            WindowEvent::MouseWheel { device_id, delta, phase } => {
                // ignore the pixel delta for now
                match delta {
                    MouseScrollDelta::LineDelta(cols, rows) => {
                        self.scroll_delta += rows;
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                        self.scroll_delta += pos.y as f32 / 100.0;
                    }
                }
                true
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

    fn peek_scroll_delta(&self) -> f32 {
        self.scroll_delta
    }

    fn consume_scroll_delta(&mut self) -> f32 {
        let scroll_delta = self.scroll_delta;
        self.scroll_delta = 0.0;
        scroll_delta
    }
}

struct CameraController {
    input: CameraInput, 
    translation_speed: f32,
    rotation_speed: f32
}

impl CameraController {
    const MIN_SPEED: f32 = 0.0001;
    const MAX_SPEED: f32 = 10000.0;
    const SPEED_MULT: f32 = 1.1;

    fn new(translation_speed: f32, rotation_speed: f32) -> CameraController {
        CameraController {
            input: CameraInput::new(),
            translation_speed,
            rotation_speed,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        self.input.process_events(event)
    }

    // update the position of the camera based on inputs and collisions with the planets
    fn update_camera(&mut self, camera: &mut Camera, planet_sim: &PlanetSim) {
        // update camera speed based on scroll
        self.translation_speed = (self.translation_speed * Self::SPEED_MULT.powf(self.input.consume_scroll_delta())).clamp(Self::MIN_SPEED, Self::MAX_SPEED);

        // update the position
        // have to perform a change of basis to get to camera space
        let translation_vector = self.translation_speed * Vector4::new(
            self.input.move_right_input(), 
            0.0,
            self.input.move_forward_input(),
            1.0,
        );

        // i personally want the up and down not to depend on camera orientation, but maybe ill make that a toggle
        let yaw = Matrix4::from_angle_y(Rad(camera.yaw));
        camera.pos += (yaw * translation_vector).xyz();
        camera.pos.y += self.input.move_up_input() * self.translation_speed;

        // handle collisions
        for obj in &planet_sim.objects {
            let obj_pos = obj.get_low_precision_position(planet_sim.render_scale);
            let radius_in_camera_space = (obj.radius / planet_sim.render_scale) as f32;

            // if camera is too close to the surface of the planet, push it back out along the vector between it and the planet
            // "too close" being around 3 * znear above the surface, and less and the camera clips
            let closest_dist = radius_in_camera_space + 3.0 * camera.znear;
            if camera.pos.distance(obj_pos) < closest_dist {
                let dir = {
                    if camera.pos - obj_pos == Vector3::zero() {
                        Vector3::unit_z()
                    } else {
                        (camera.pos - obj_pos).normalize()
                    }
                };
                camera.pos = obj_pos + closest_dist * dir;
            }   
        }

        // update the rotation
        camera.yaw += self.input.turn_right_input() * self.rotation_speed;
        camera.yaw %= 2.0 * f32::consts::PI;
        camera.pitch += self.input.turn_up_input() * self.rotation_speed;
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
    window: &'a Window,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface<'a>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    sphere_mesh: model::Mesh,
    planet_textures: model::Material,
    planet_instance_buffer: wgpu::Buffer,
    skysphere_instance: AstroBody,
    skysphere_instance_buffer: wgpu::Buffer,
    point_vertices: Vec<PointVertex>,
    point_vertex_buffer: Buffer,
    // diffuse_bind_group: wgpu::BindGroup,
    // diffuse_texture: texture::Texture,
    depth_texture: texture::Texture,
    render_pipelines: Vec<wgpu::RenderPipeline>,
    planet_sim: Arc<Mutex<PlanetSim>>,
    frame_counter: FrameCounter,
    sec_per_sec: Arc<Mutex<u32>>,
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
        

        // camera initialization
        let (camera, camera_controller, camera_uniform, camera_bind_group_layout, camera_bind_group, camera_buffer) = 
            Self::new_camera(&device, &config, &planet_sim);
        

        // load the planet texture array
        let texture_array = resources::load_texture_array(
            &[
                "textures/2k_stars_milky_way.jpg",
                "textures/2k_sun.jpg",   
                "textures/2k_mercury.jpg",  
                "textures/2k_venus_atmosphere.jpg",
                "textures/2k_earth_daymap.jpg", 
                "textures/2k_moon.jpg",
                "textures/2k_mars.jpg", 
                "textures/2k_jupiter.jpg", 
                "textures/2k_saturn.jpg", 
                "textures/2k_uranus.jpg", 
                "textures/2k_neptune.jpg", 
            ], 
            &device, &queue, "planet_texture_array"
        ).await.unwrap();
        
        let texture_array_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
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
                label: Some("texture_array_bind_group_layout"),
            });
            
        // create the bind group using the above layout
        let texture_array_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_array_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_array.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&texture_array.sampler),
                    }
                ],
                label: Some("texture_array_bind_group"),
            }
        );
        
        // create the instance buffer for the astronomical bodies
        let planet_instance_data = planet_sim.instance_data();
        let planet_instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("AstroBody Instance Buffer"),
                contents: bytemuck::cast_slice(&planet_instance_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        // create the instance buffer for the single skybox
        let skysphere_instance = AstroBody {
            label: String::from("Skybox"),
            texture_index: 0,
            radius: camera.zfar as f64,
            position: camera.get_high_precision_position(),
            ..Default::default()
        };
        let skysphere_instance_data = vec![skysphere_instance.to_raw_instance(1.0)];
        let skysphere_instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("AstroBody Instance Buffer"),
                contents: bytemuck::cast_slice(&skysphere_instance_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        // create the point vertices
        let point_vertices = planet_sim.objects
            .iter()
            .map(|obj| point_vertex_from_astro_body(obj, planet_sim.render_scale))
            .collect::<Vec<PointVertex>>();
        // create the point vertex buffer
        let point_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Vertex Buffer"),
            contents: bytemuck::cast_slice(&point_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        
        // load the sphere mesh
        let sphere_mesh = resources::load_mesh("sphere_mesh_2.obj", &device, &queue, &texture_array_bind_group_layout)
            .await.unwrap();
        // store the texture array in a material
        let planet_textures = model::Material {
            name: String::from("PlanetTextureMaterial"),
            diffuse_texture: texture_array,
            bind_group: texture_array_bind_group,
        };

        // create a depth texture
        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");




        // main shader which renders the planets, skybox, etc
        let main_shader = device.create_shader_module(wgpu::include_wgsl!("main_shader.wgsl"));        

        // the layout for the pipeline, useful for hotswapping pipelines i think
        let main_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Main Render Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &texture_array_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        // create the render pipeline
        let main_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Main Render Pipeline"),
            layout: Some(&main_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &main_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    model::ModelVertex::desc(),
                    AstroBodyInstanceRaw::desc(),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
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
                // cull_mode: Some(wgpu::Face::Back),
                cull_mode: None,
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



        // the point shader specifically renders points at the centers of the planets so that they can be seen from far away
        let point_shader = device.create_shader_module(wgpu::include_wgsl!("point_shader.wgsl"));        

        // the layout for the pipeline, useful for hotswapping pipelines i think
        let point_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Point Render Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        // create the render pipeline
        let point_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Point Render Pipeline"),
            layout: Some(&point_render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &point_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    PointVertex::desc(),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &point_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // cull_mode: Some(wgpu::Face::Back),
                cull_mode: None,
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
            // depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None, 
        });

        // create the state
        State {
            window,
            size,
            surface,
            config,
            device,
            queue,
            camera,
            camera_controller,
            camera_uniform,
            camera_bind_group,
            camera_buffer,
            sphere_mesh,
            planet_textures,
            planet_instance_buffer,
            skysphere_instance,
            skysphere_instance_buffer,
            point_vertices,
            point_vertex_buffer,
            // diffuse_bind_group: wgpu::BindGroup,
            // diffuse_texture: texture::Texture,
            depth_texture,
            render_pipelines: vec![main_render_pipeline, point_render_pipeline],
            planet_sim: Arc::new(Mutex::new(planet_sim)), 
            sec_per_sec: Arc::new(Mutex::new(1)),
            frame_counter: FrameCounter::new(),
        }
    }

    // initialize the camera, doesn't really need to be in the main initialization
    fn new_camera(device: &Device, config: &SurfaceConfiguration, planet_sim: &PlanetSim) 
        -> (Camera, CameraController, CameraUniform, BindGroupLayout, BindGroup, Buffer) 
    {
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
            znear: 0.01,
            zfar: 1000000000.0,
        };

        let camera_controller = CameraController::new(0.003, 0.02);

        // create the camera uniform and buffer
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

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

        (camera, camera_controller, camera_uniform, camera_bind_group_layout, camera_bind_group, camera_buffer)
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

        // update the camera
        self.camera.aspect = self.config.width as f32 / self.config.height as f32;
        {
            let ps = self.planet_sim.lock().unwrap();
            self.camera_controller.update_camera(&mut self.camera, &ps);
        }
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));

        // update skysphere position
        self.skysphere_instance.position = self.camera.get_high_precision_position();
        self.queue.write_buffer(&self.skysphere_instance_buffer, 0, bytemuck::cast_slice(&vec![self.skysphere_instance.to_raw_instance(1.0)]));

        // update the planet instances
        self.planet_sim.lock().unwrap().update(*self.sec_per_sec.lock().unwrap() * self.frame_counter.clamped_delta_time());
        let planet_instance_data = self.planet_sim.lock().unwrap().instance_data();
        self.queue.write_buffer(&self.planet_instance_buffer, 0, bytemuck::cast_slice(&planet_instance_data));

        // update point vertex positions
        {
            let ps = self.planet_sim.lock().unwrap();
            self.point_vertices = ps.objects
                .iter()
                .map(|obj| point_vertex_from_astro_body(obj, ps.render_scale))
                .collect::<Vec<PointVertex>>();
        }
        self.queue.write_buffer(&self.point_vertex_buffer, 0, bytemuck::cast_slice(&self.point_vertices));
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
            let mut point_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(
                                wgpu::Color::BLACK,
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
                // depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // point render pass
            point_render_pass.set_pipeline(&self.render_pipelines[1]); 
            point_render_pass.set_vertex_buffer(0, self.point_vertex_buffer.slice(..));
            point_render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            point_render_pass.draw(0..self.point_vertices.len() as u32, 0..1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }
                    })
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // render pass
            render_pass.set_pipeline(&self.render_pipelines[0]); 
            
            // setup sphere mesh vertices and bind groups
            render_pass.set_vertex_buffer(0, self.sphere_mesh.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.sphere_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.planet_textures.bind_group, &[]);
            
            // render skysphere
            render_pass.set_vertex_buffer(1, self.skysphere_instance_buffer.slice(..));
            render_pass.draw_indexed(0..self.sphere_mesh.num_elements, 0, 0..1);
            
            // draw the instanced planet meshes
            render_pass.set_vertex_buffer(1, self.planet_instance_buffer.slice(..));
            let instances = self.planet_sim.lock().unwrap().len() as u32;
            render_pass.draw_indexed(0..self.sphere_mesh.num_elements, 0, 0..instances);
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
    let mut planet_sim = PlanetSim::new(500.0, 10.0);
    planet_sim.add(AstroBody {
        label: String::from("Sun"),
        texture_index: 1,
        color: [1.0, 1.0, 1.0, 1.0],
        mass: 1000.0, 
        radius: 5.0, 
        position: Vector3::new(0.0, 0.0, 0.0),
        velocity: Vector3::new(0.0, 0.0, 0.0),
        rotation: Quaternion::zero(),
        axis_of_rotation: Vector3::unit_z(),
        angular_velocity: 0.0,
    });
    planet_sim.add(AstroBody {
        label: String::from("Mars"),
        texture_index: 6,
        color: [0.89, 0.471, 0.259, 1.0],
        mass: 10.0, 
        radius: 1.0, 
        position: Vector3::new(-50.0, 0.0, 0.0),
        velocity: Vector3::new(0.0, 0.0, -100.0),
        rotation: Quaternion::zero(),
        axis_of_rotation: Vector3::unit_z(),
        angular_velocity: 0.0,
    });
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
    planet_sim.set_focused(Some("Sun"));

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

