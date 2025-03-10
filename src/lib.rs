// lib.rs

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

use wgpu::util::DeviceExt;



#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    // get the vertexbufferlayout associated with the Vertex struct
    // the vertexbufferlayout is just how each vertex is laid out within the buffer
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            // the attributes specify how each vertex should be further divided
            attributes: &[
                // this attribute corresponds to the position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // and this attribute, which corresponds to the color, is offset by the size of the previous attribute
                // the shader location lets wgsl know what @location(i) refers to, so in this case @location(1) refers to the vec3f field for color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }
            ]
            // could use the following wgpu macro to auto-generate the attributes
            // attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        }
    }
}

// a simple pentagon
const VERTICES: &[Vertex] = &[
    Vertex { position: [-0.0868241, 0.49240386, 0.0], color: [1.0, 0.0, 0.0] }, // A
    Vertex { position: [-0.49513406, 0.06958647, 0.0], color: [0.4, 0.6, 0.0] }, // B
    Vertex { position: [-0.21918549, -0.44939706, 0.0], color: [0.0, 0.8, 0.2] }, // C
    Vertex { position: [0.35966998, -0.3473291, 0.0], color: [0.0, 0.2, 0.8] }, // D
    Vertex { position: [0.44147372, 0.2347359, 0.0], color: [0.4, 0.0, 0.6] }, // E
];

// vertex buffer indices
const INDICES: &[u16] = &[
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
];


// webgpu state including the surface, device, and render pipeline
struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: &'a Window,
    render_pipeline: wgpu::RenderPipeline,
    num_vertices: u32,
    vertex_buffer: wgpu::Buffer,
    num_indices: u32,
    index_buffer: wgpu::Buffer, 
}


impl<'a> State<'a> {
    // Creating some of the wgpu types requires async code
    async fn new(window: &'a Window) -> State<'a> {
        let size = window.inner_size();

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

        // 
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
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
                    Vertex::desc(),
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
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None, 
        });

        // want to store the number of vertices for rendering
        let num_vertices = VERTICES.len() as u32;
        // create a vertex buffer for, you guessed it, the vertices
        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        // number of indices
        let num_indices = INDICES.len() as u32;
        // index buffer to reuse the vertices
        // though an index buffer still has some waste, it wastes 2 bytes per vertex instead of sizeof(vec3f) = 12 bytes per vertex
        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        // create the state
        State {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            num_vertices,
            vertex_buffer,
            num_indices,
            index_buffer,
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
            self.surface.configure(&self.device, &self.config);
        }
    }

    // for handling inputs, currently unused
    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    // update the state
    fn update(&mut self) {
        
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
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline); 
            // need to actually assign the buffer we created to the renderer (since it needs a specific slot)
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            // as with the vertex buffer, we have to set the active index buffer, but this time there is only one slot (hence active)
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16); 
            // instead of using draw, since we are using an index buffer, we have to use draw_indexed
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
            
            // draw 1 instance of all of our vertices
            // render_pass.draw(0..self.num_vertices, 0..1);   
            // render_pass.draw(0..3, 0..1); 
            
        }
        
    
        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    
        Ok(())
    }
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
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    // if targetting wasm, add the canvas that is displaying the webgl to the document
    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        let _ = window.request_inner_size(PhysicalSize::new(640, 640));
        
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("planet-sim")?;
                let canvas = web_sys::Element::from(window.canvas()?);
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    } 

    // create the state
    let mut state = State::new(&window).await;
    let mut surface_configured = false;

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

