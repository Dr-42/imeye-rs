use std::ffi::CString;
use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};
use std::io::Cursor;

use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextAttributesBuilder, NotCurrentGlContext};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::GlSurface;
use glutin_winit::{DisplayBuilder, GlWindow};

use raw_window_handle::HasRawWindowHandle;
use winit::event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowBuilder;

// --- DEDICATED GPU SELECTION (WINDOWS) ---
#[cfg(target_os = "windows")]
#[allow(non_upper_case_globals)]
#[no_mangle]
pub static NvOptimusEnablement: u32 = 1;

#[cfg(target_os = "windows")]
#[allow(non_upper_case_globals)]
#[no_mangle]
pub static AmdPowerXpressRequestHighPerformance: u32 = 1;
// -----------------------------------------
// --- DEDICATED GPU SELECTION (LINUX) ---
#[cfg(target_os = "linux")]
#[allow(non_upper_case_globals)]
#[no_mangle]
pub static __NV_PRIME_RENDER_OFFLOAD: u32 = 1;

#[cfg(target_os = "linux")]
#[allow(non_upper_case_globals)]
#[no_mangle]
pub static __GLX_VENDOR_LIBRARY_NAME: u32 = 1;
// -----------------------------------------

const VERT_SHADER_SRC: &str = r#"
    #version 330 core
    layout (location = 0) in vec3 vertex;
    layout (location = 1) in vec2 texCoord;
    out vec2 TexCoords;
    uniform float rotation_angle;
    void main() {
       float rad = radians(rotation_angle);
       mat4 rotation = mat4(
           cos(rad), sin(rad), 0.0, 0.0,
           -sin(rad), cos(rad), 0.0, 0.0,
           0.0, 0.0, 1.0, 0.0,
           0.0, 0.0, 0.0, 1.0
       );
       vec4 pos = rotation * vec4(vertex, 1.0);
       TexCoords = texCoord;
       gl_Position = pos;
    }
"#;

// Updated Fragment Shader for a "Material Design" style spinner
const FRAG_SHADER_SRC: &str = r#"
    #version 330 core
    in vec2 TexCoords;
    out vec4 color;
    uniform sampler2D image;
    uniform int is_spinner; 
    
    void main() {   
        if (is_spinner == 1) {
            // Normalize UVs to -0.5 -> 0.5 range for centering
            vec2 uv = TexCoords - 0.5;
            float dist = length(uv);
            
            // Calculate angle for the comet tail effect (-PI to PI)
            float angle = atan(uv.y, uv.x);
            
            // Map angle to 0.0 -> 1.0 range
            float val = (angle + 3.14159) / 6.28318;
            
            // Create the ring shape (Radius ~0.35, Thickness ~0.1)
            // Use smoothstep for nice anti-aliased edges
            float ring = smoothstep(0.25, 0.30, dist) * (1.0 - smoothstep(0.40, 0.45, dist));
            
            // Apply gradient: val^2 makes the tail fade out nicely (comet look)
            float brightness = pow(val, 2.0);
            
            // Combine ring shape with gradient brightness
            // Color: Bright Blue-ish (0.2, 0.6, 1.0)
            color = vec4(0.2, 0.6, 1.0, ring * brightness);
        } else {
            vec4 texColor = texture(image, TexCoords);
            if (texColor.a < 0.1)
                discard;
            color = texColor;
        }
    }
"#;

// OPTIMIZATION: Texture Coordinates flipped here (V: 0.0 <-> 1.0)
// This avoids the expensive CPU-side image.flipv() operation.
const VERTICES: [f32; 20] = [
    // Position         // Tex coords (V flipped)
    1.0, 1.0, -1.0, 1.0, 0.0, // Top Right
    1.0, -1.0, -1.0, 1.0, 1.0, // Bottom Right
    -1.0, -1.0, -1.0, 0.0, 1.0, // Bottom Left
    -1.0, 1.0, -1.0, 0.0, 0.0, // Top Left
];
const INDICES: [u32; 6] = [0, 3, 1, 1, 3, 2];

struct LoadedImage {
    pixels: Vec<u8>,
    width: i32,
    height: i32,
    generation_id: u64,
}

struct AppState {
    shader_program: u32,
    #[allow(dead_code)]
    vao: u32,
    texture: u32,

    // Dimensions
    base_width: f32,
    base_height: f32,

    // Viewport State
    view_x: f32,
    view_y: f32,
    rotation: i32,

    // Zoom Math
    zoom_level: f32,
    target_zoom: f32,
    zoom_focus: (f32, f32),

    // Mouse
    is_dragging: bool,
    last_mouse_pos: Option<(f64, f64)>,
    mouse_pos: (f64, f64),

    // Navigation & Async Loading
    image_list: Vec<PathBuf>,
    image_index: usize,
    monitor_size: (u32, u32),

    // Async State
    loading: bool,
    load_generation: u64,
    rx: Receiver<LoadedImage>,
    tx: Sender<LoadedImage>,
    spinner_rotation: f32,
}

impl AppState {
    fn new(filepath: PathBuf, monitor_size: (u32, u32)) -> Self {
        let (image_list, image_index) = Self::scan_directory(&filepath);
        let (shader_program, vao, texture, w, h) = unsafe { Self::init_gl(&filepath) };
        let (tx, rx) = channel();

        let mut state = Self {
            shader_program,
            vao,
            texture,
            base_width: w as f32,
            base_height: h as f32,
            view_x: 0.0,
            view_y: 0.0,
            rotation: 0,
            zoom_level: 1.0,
            target_zoom: 1.0,
            zoom_focus: (0.0, 0.0),
            is_dragging: false,
            last_mouse_pos: None,
            mouse_pos: (0.0, 0.0),
            image_list,
            image_index,
            monitor_size,
            loading: false,
            load_generation: 0,
            rx,
            tx,
            spinner_rotation: 0.0,
        };
        state.reset_view();
        state
    }

    fn lerp(start: f32, end: f32, t: f32) -> f32 {
        start + (end - start) * t
    }

    fn update_smoothness(&mut self) {
        if (self.target_zoom - self.zoom_level).abs() > 0.001 {
            let old_zoom = self.zoom_level;
            self.zoom_level = Self::lerp(self.zoom_level, self.target_zoom, 0.15);
            let new_zoom = self.zoom_level;

            // Invariant Point Zoom Math
            let focus_x = self.zoom_focus.0;
            let focus_y = self.zoom_focus.1;
            let ratio = new_zoom / old_zoom;
            let offset_x = focus_x - self.view_x;
            let offset_y = focus_y - self.view_y;

            self.view_x = focus_x - (offset_x * ratio);
            self.view_y = focus_y - (offset_y * ratio);
        }

        if self.loading {
            // 8.0 degrees per frame at 60fps is smooth and not too fast
            self.spinner_rotation = (self.spinner_rotation + 8.0) % 360.0;
        }
    }

    fn scan_directory(target: &Path) -> (Vec<PathBuf>, usize) {
        let parent = target.parent().unwrap_or(Path::new("."));
        let mut images = Vec::new();
        let valid_exts = [
            "png", "jpg", "jpeg", "bmp", "tga", "gif", "hdr", "pic", "pnm", "arw",
        ];

        if let Ok(entries) = fs::read_dir(parent) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if let Some(ext_str) = ext.to_str() {
                        if valid_exts.contains(&ext_str.to_lowercase().as_str()) {
                            images.push(path);
                        }
                    }
                }
            }
        }
        images.sort();
        let index = images.iter().position(|p| p == target).unwrap_or(0);
        (images, index)
    }

    unsafe fn init_gl(path: &Path) -> (u32, u32, u32, i32, i32) {
        let vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
        let c_str_vert = CString::new(VERT_SHADER_SRC.as_bytes()).unwrap();
        gl::ShaderSource(vertex_shader, 1, &c_str_vert.as_ptr(), std::ptr::null());
        gl::CompileShader(vertex_shader);

        let fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
        let c_str_frag = CString::new(FRAG_SHADER_SRC.as_bytes()).unwrap();
        gl::ShaderSource(fragment_shader, 1, &c_str_frag.as_ptr(), std::ptr::null());
        gl::CompileShader(fragment_shader);

        let shader_program = gl::CreateProgram();
        gl::AttachShader(shader_program, vertex_shader);
        gl::AttachShader(shader_program, fragment_shader);
        gl::LinkProgram(shader_program);
        gl::DeleteShader(vertex_shader);
        gl::DeleteShader(fragment_shader);

        // Enable Alpha Blending for the spinner's smooth transparency
        gl::Enable(gl::BLEND);
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        gl::ClearColor(0.0, 0.0, 0.0, 1.0);

        let (mut vao, mut vbo, mut ebo) = (0, 0, 0);
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);
        gl::GenBuffers(1, &mut ebo);

        gl::BindVertexArray(vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (VERTICES.len() * 4) as isize,
            VERTICES.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );
        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
        gl::BufferData(
            gl::ELEMENT_ARRAY_BUFFER,
            (INDICES.len() * 4) as isize,
            INDICES.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );

        let pos_attrib =
            gl::GetAttribLocation(shader_program, CString::new("vertex").unwrap().as_ptr());
        gl::VertexAttribPointer(
            pos_attrib as u32,
            3,
            gl::FLOAT,
            gl::FALSE,
            5 * 4,
            std::ptr::null(),
        );
        gl::EnableVertexAttribArray(pos_attrib as u32);

        let tex_attrib =
            gl::GetAttribLocation(shader_program, CString::new("texCoord").unwrap().as_ptr());
        gl::VertexAttribPointer(
            tex_attrib as u32,
            2,
            gl::FLOAT,
            gl::FALSE,
            5 * 4,
            (3 * 4) as *const _,
        );
        gl::EnableVertexAttribArray(tex_attrib as u32);

        // Initial sync load
        let (tex_id, w, h) = Self::load_texture_sync(path);
        (shader_program, vao, tex_id, w, h)
    }

unsafe fn load_texture_sync(path: &Path) -> (u32, i32, i32) {
        let img = rawler::decode_file(path).expect("Failed to load image");
        
        let u16_pixels = img.pixels_u16();
        
        // Convert u16 pixels to little-endian bytes properly for bayer Depth16LE
        let mut img_bytes = Vec::with_capacity(u16_pixels.len() * 2);
        for &p in u16_pixels {
            img_bytes.extend_from_slice(&p.to_le_bytes());
        }
        
        let (w, h) = (img.width as usize, img.height as usize);
        
        let depth = bayer::BayerDepth::Depth16LE;
        let cfa = bayer::CFA::RGGB;

        // Since input is Depth16LE, output must be Depth16.
        // Depth16 uses 6 bytes per pixel (RGB, 2 bytes per channel).
        let mut buf = vec![0u8; w * h * 6];

        let mut dst = bayer::RasterMut::new(
            w, h, bayer::RasterDepth::Depth16,
            &mut buf
        );
        
        bayer::run_demosaic(
            &mut Cursor::new(&img_bytes[..]), 
            depth, 
            cfa, 
            bayer::Demosaic::Linear, 
            &mut dst
        ).expect("Failed to run demosaic");
        
        // Find the maximum pixel value for normalization. 
        // RAW files are often 12-bit or 14-bit, not full 16-bit.
        // Without scaling, the image will appear extremely dark.
        let mut max_val = 1;
        for chunk in buf.chunks_exact(2) {
            let val = u16::from_ne_bytes([chunk[0], chunk[1]]);
            if val > max_val { 
                max_val = val; 
            }
        }
        let scale = 255.0 / (max_val as f32);

        // Downsample the 16-bit RGB buffer to 8-bit RGBA for OpenGL upload
        let mut rgba_buf = Vec::with_capacity(w * h * 4);
        for chunk in buf.chunks_exact(6) {
            let r = u16::from_ne_bytes([chunk[0], chunk[1]]);
            let g = u16::from_ne_bytes([chunk[2], chunk[3]]);
            let b = u16::from_ne_bytes([chunk[4], chunk[5]]);
            
            rgba_buf.push(((r as f32) * scale).clamp(0.0, 255.0) as u8);
            rgba_buf.push(((g as f32) * scale).clamp(0.0, 255.0) as u8);
            rgba_buf.push(((b as f32) * scale).clamp(0.0, 255.0) as u8);
            rgba_buf.push(255); // A
        }

        Self::upload_texture(&rgba_buf, w as i32, h as i32)
    }

    unsafe fn upload_texture(data: &[u8], w: i32, h: i32) -> (u32, i32, i32) {
        let mut texture = 0;
        gl::GenTextures(1, &mut texture);
        gl::BindTexture(gl::TEXTURE_2D, texture);

        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);
        // OPTIMIZATION: Use GL_LINEAR instead of MIPMAP to avoid blocking generation
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

        gl::TexImage2D(
            gl::TEXTURE_2D,
            0,
            gl::RGBA as i32,
            w,
            h,
            0,
            gl::RGBA,
            gl::UNSIGNED_BYTE,
            data.as_ptr() as *const _,
        );
        // OPTIMIZATION: gl::GenerateMipmap removed

        (texture, w, h)
    }

    fn reset_view(&mut self) {
        let (disp_w, disp_h) = (self.monitor_size.0 as f32, self.monitor_size.1 as f32);
        let mut scale = 1.0;
        if self.base_width > disp_w {
            scale = disp_w / self.base_width;
        }
        if self.base_height > disp_h {
            let h_scale = disp_h / self.base_height;
            if h_scale < scale {
                scale = h_scale;
            }
        }
        scale *= 0.95;

        self.zoom_level = scale;
        self.target_zoom = scale;

        let current_w = self.base_width * self.zoom_level;
        let current_h = self.base_height * self.zoom_level;

        self.view_x = (disp_w - current_w) / 2.0;
        self.view_y = (disp_h - current_h) / 2.0;
        self.rotation = 0;
    }

    fn trigger_load_next(&mut self, direction: i32) {
        if self.image_list.is_empty() {
            return;
        }

        let len = self.image_list.len();
        if direction > 0 {
            self.image_index = (self.image_index + 1) % len;
        } else {
            if self.image_index == 0 {
                self.image_index = len - 1;
            } else {
                self.image_index -= 1;
            }
        }

        self.loading = true;
        self.load_generation += 1;
        let gen_id = self.load_generation;
        let path = self.image_list[self.image_index].clone();
        let tx = self.tx.clone();

        thread::spawn(move || {
            if let Ok(img) = image::open(&path) {
                // OPTIMIZATION: removed .flipv()
                let (w, h) = (img.width() as i32, img.height() as i32);
                let pixels = img.to_rgba8().into_raw();

                let _ = tx.send(LoadedImage {
                    pixels,
                    width: w,
                    height: h,
                    generation_id: gen_id,
                });
            }
        });
    }

    fn finalize_load(&mut self, image: LoadedImage) {
        if image.generation_id != self.load_generation {
            return;
        }

        self.loading = false;
        unsafe {
            let prev_width = self.base_width;
            let prev_height = self.base_height;
            let prev_zoom = self.zoom_level;

            let (new_tex, w, h) = Self::upload_texture(&image.pixels, image.width, image.height);
            gl::DeleteTextures(1, &self.texture);
            self.texture = new_tex;
            self.base_width = w as f32;
            self.base_height = h as f32;

            // Maintain perceived scale
            let prev_area = prev_width * prev_height;
            let new_area = self.base_width * self.base_height;
            let area_ratio = if new_area > 0.0 {
                (prev_area.sqrt()) / (new_area.sqrt())
            } else {
                1.0
            };

            self.zoom_level *= area_ratio;
            self.target_zoom = self.zoom_level;

            // Align Centers
            let prev_center_x = self.view_x + (prev_width * prev_zoom) / 2.0;
            let prev_center_y = self.view_y + (prev_height * prev_zoom) / 2.0;

            let new_w = self.base_width * self.zoom_level;
            let new_h = self.base_height * self.zoom_level;

            self.view_x = prev_center_x - new_w / 2.0;
            self.view_y = prev_center_y - new_h / 2.0;

            self.rotation = 0;
        }
    }

    fn rotate(&mut self, dir: i32) {
        self.rotation += dir * 90;
        std::mem::swap(&mut self.base_width, &mut self.base_height);
        let current_w = self.base_width * self.zoom_level;
        let current_h = self.base_height * self.zoom_level;
        self.view_x = self.view_x - (current_w - current_h) / 2.0;
        self.view_y = self.view_y - (current_h - current_w) / 2.0;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: imeye_rs <filename>");
        return;
    }
    let filepath = PathBuf::from(&args[1]);

    let event_loop = EventLoop::new().unwrap_or_else(|e| {
        eprintln!("Error: Failed to create event loop: {}", e);
        eprintln!("\nThis often happens on Linux when windowing system libraries are missing or incompatible.");
        eprintln!("Ensure you have the following libraries installed:");
        eprintln!("  - Wayland: libwayland-client, libxkbcommon");
        eprintln!("  - X11: libX11, libxkbcommon");
        eprintln!("\nIf you are using Nix, ensure they are available in your shell environment.");
        std::process::exit(1);
    });
    let window_builder = WindowBuilder::new()
        .with_title("imeye-rs")
        .with_visible(true);
    let template = ConfigTemplateBuilder::new()
        .with_alpha_size(8)
        .with_transparency(false);
    let display_builder = DisplayBuilder::new().with_window_builder(Some(window_builder));

    let (window, gl_config) = display_builder
        .build(&event_loop, template, |configs| {
            configs
                .reduce(|accum, config| {
                    if config.num_samples() > accum.num_samples() {
                        config
                    } else {
                        accum
                    }
                })
                .expect("Failed to find a suitable OpenGL configuration.")
        })
        .expect("Failed to build the display. Ensure OpenGL drivers are installed.");

    let window = window.expect("Failed to create the window.");
    let raw_window_handle = window.raw_window_handle();
    let gl_display = gl_config.display();
    let context_attributes = ContextAttributesBuilder::new().build(Some(raw_window_handle));

    let not_current_gl_context = unsafe {
        gl_display
            .create_context(&gl_config, &context_attributes)
            .expect("failed to create context")
    };

    let attrs = window.build_surface_attributes(Default::default());
    let gl_surface = unsafe {
        gl_display
            .create_window_surface(&gl_config, &attrs)
            .expect("Failed to create the OpenGL window surface.")
    };
    let gl_context = not_current_gl_context.make_current(&gl_surface).expect("Failed to make the OpenGL context current.");

    gl::load_with(|symbol| gl_display.get_proc_address(&CString::new(symbol).expect("Failed to create CString for GL symbol")));

    let monitor_size = if let Some(monitor) = window.current_monitor() {
        (monitor.size().width, monitor.size().height)
    } else {
        (1920, 1080)
    };

    let mut app_state = AppState::new(filepath, monitor_size);
    let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(
        (app_state.base_width * app_state.zoom_level) as u32,
        (app_state.base_height * app_state.zoom_level) as u32,
    ));

    let mut last_frame = Instant::now();

    let _ = event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        if let Ok(loaded_image) = app_state.rx.try_recv() {
            app_state.finalize_load(loaded_image);
            window.set_title(&format!(
                "imeye - {:?}",
                app_state.image_list[app_state.image_index]
            ));
            window.request_redraw();
        }

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),

                WindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                } => {
                    app_state.is_dragging = state == ElementState::Pressed;
                }
                WindowEvent::CursorMoved { position, .. } => {
                    app_state.mouse_pos = (position.x, position.y);

                    if app_state.is_dragging {
                        if let Some(last_pos) = app_state.last_mouse_pos {
                            let dx = position.x - last_pos.0;
                            let dy = position.y - last_pos.1;
                            app_state.view_x += dx as f32;
                            app_state.view_y -= dy as f32;
                        }
                    }
                    app_state.last_mouse_pos = Some((position.x, position.y));
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    app_state.zoom_focus =
                        (app_state.mouse_pos.0 as f32, app_state.mouse_pos.1 as f32);

                    let scroll_amount = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                        MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) * 0.001,
                    };
                    app_state.target_zoom += scroll_amount;
                    if app_state.target_zoom < 0.01 {
                        app_state.target_zoom = 0.01;
                    }
                }

                WindowEvent::RedrawRequested => {
                    app_state.update_smoothness();

                    unsafe {
                        gl::Clear(gl::COLOR_BUFFER_BIT);
                        gl::UseProgram(app_state.shader_program);

                        let rot_loc = gl::GetUniformLocation(
                            app_state.shader_program,
                            CString::new("rotation_angle").unwrap().as_ptr(),
                        );
                        let spinner_loc = gl::GetUniformLocation(
                            app_state.shader_program,
                            CString::new("is_spinner").unwrap().as_ptr(),
                        );

                        gl::Uniform1i(spinner_loc, 0);
                        gl::Uniform1f(rot_loc, app_state.rotation as f32);

                        let w = (app_state.base_width * app_state.zoom_level) as i32;
                        let h = (app_state.base_height * app_state.zoom_level) as i32;
                        gl::Viewport(app_state.view_x as i32, app_state.view_y as i32, w, h);
                        gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, std::ptr::null());

                        if app_state.loading {
                            gl::Uniform1i(spinner_loc, 1);
                            gl::Uniform1f(rot_loc, app_state.spinner_rotation);

                            let win_size = window.inner_size();
                            let spinner_size = 50;
                            let center_x = (win_size.width as i32 - spinner_size) / 2;
                            let center_y = (win_size.height as i32 - spinner_size) / 2;

                            gl::Viewport(center_x, center_y, spinner_size, spinner_size);
                            gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, std::ptr::null());
                        }

                        window.pre_present_notify();
                        gl_surface.swap_buffers(&gl_context).unwrap();
                    }
                }

                WindowEvent::Resized(physical_size) => {
                    if physical_size.width > 0 && physical_size.height > 0 {
                        gl_surface.resize(
                            &gl_context,
                            NonZeroU32::new(physical_size.width).unwrap(),
                            NonZeroU32::new(physical_size.height).unwrap(),
                        );
                    }
                    let current_w = app_state.base_width * app_state.zoom_level;
                    let current_h = app_state.base_height * app_state.zoom_level;
                    app_state.view_x = (physical_size.width as f32 - current_w) / 2.0;
                    app_state.view_y = (physical_size.height as f32 - current_h) / 2.0;
                    app_state.monitor_size = (physical_size.width, physical_size.height);
                }

                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(code),
                            state,
                            ..
                        },
                    ..
                } => {
                    if state == ElementState::Pressed {
                        match code {
                            KeyCode::Escape => elwt.exit(),
                            KeyCode::KeyF => {
                                if window.fullscreen().is_some() {
                                    window.set_fullscreen(None);
                                } else {
                                    window.set_fullscreen(Some(
                                        winit::window::Fullscreen::Borderless(
                                            window.current_monitor(),
                                        ),
                                    ));
                                }
                            }
                            KeyCode::KeyR => app_state.reset_view(),
                            KeyCode::KeyQ => app_state.rotate(1),
                            KeyCode::KeyE => app_state.rotate(-1),

                            KeyCode::ArrowRight => app_state.trigger_load_next(1),
                            KeyCode::ArrowLeft => app_state.trigger_load_next(-1),

                            KeyCode::ArrowUp => {
                                let win_size = window.inner_size();
                                app_state.zoom_focus =
                                    (win_size.width as f32 / 2.0, win_size.height as f32 / 2.0);
                                app_state.target_zoom += 0.1;
                            }
                            KeyCode::ArrowDown => {
                                let win_size = window.inner_size();
                                app_state.zoom_focus =
                                    (win_size.width as f32 / 2.0, win_size.height as f32 / 2.0);
                                app_state.target_zoom -= 0.1;
                                if app_state.target_zoom < 0.01 {
                                    app_state.target_zoom = 0.01;
                                }
                            }

                            KeyCode::KeyW => app_state.view_y += 20.0,
                            KeyCode::KeyS => app_state.view_y -= 20.0,
                            KeyCode::KeyA => app_state.view_x -= 20.0,
                            KeyCode::KeyD => app_state.view_x += 20.0,
                            _ => {}
                        }
                    }
                }
                _ => (),
            },
            Event::AboutToWait => {
                let wait_time = if app_state.loading { 8 } else { 16 };
                if last_frame.elapsed() >= Duration::from_millis(wait_time) {
                    window.request_redraw();
                    last_frame = Instant::now();
                }
            }
            _ => (),
        }
    });
}
