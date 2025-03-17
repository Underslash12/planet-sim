// sim.rs

use cgmath::{InnerSpace, MetricSpace, Quaternion, Vector3, Vector4, Zero};
use web_time::{Instant, Duration};
use winit::dpi::Position;
use wgpu::vertex_attr_array;
use csv::ReaderBuilder;
use std::cmp::min;


#[derive(Debug)]
pub struct AstroBody {
    pub label: String,
    pub texture_index: u32,
    pub color: [f32; 4],
    pub mass: f64,
    pub radius: f64,
    pub position: cgmath::Vector3<f64>,
    pub velocity: cgmath::Vector3<f64>,
    pub rotation: cgmath::Quaternion<f32>,
    pub axis_of_rotation: cgmath::Vector3<f64>,
    pub angular_velocity: f64, 
}

impl Default for AstroBody {
    fn default() -> Self {
        AstroBody {
            label: String::from("default_astrobody"),
            texture_index: 0,
            color: [1.0, 1.0, 1.0, 1.0],
            mass: 1.0,
            radius: 1000.0,
            position: Vector3::unit_x(),
            velocity: Vector3::unit_x(),
            rotation: Quaternion::zero(),
            axis_of_rotation: Vector3::zero(),
            angular_velocity: 0.0,
        }
    }
}

impl AstroBody {
    pub fn new(label: &str, texture_index: u32, color: [f32; 4], mass: f64, radius: f64, position: cgmath::Vector3<f64>, velocity: cgmath::Vector3<f64>, 
        rotation: cgmath::Quaternion<f32>, axis_of_rotation: cgmath::Vector3<f64>, angular_velocity: f64
    ) -> AstroBody {
        AstroBody {
            label: String::from(label),
            texture_index,
            color,
            mass,
            radius,
            position,
            velocity,
            rotation,
            axis_of_rotation,
            angular_velocity,
        }
    } 

    pub fn get_low_precision_position(&self, scale: f64) -> Vector3<f32> {
        let scaled_pos = self.position / scale; 
        Vector3::new(scaled_pos.x as f32, scaled_pos.y as f32, scaled_pos.z as f32)
    }

    pub fn to_raw_instance(&self, scale: f64) -> AstroBodyInstanceRaw {
        let low_precision_position = self.get_low_precision_position(scale);
        let translation = cgmath::Matrix4::from_translation(low_precision_position);
        let scale = cgmath::Matrix4::from_scale((self.radius / scale) as f32);
        AstroBodyInstanceRaw {
            mat: (translation * scale).into(),
            tex: self.texture_index,
        }
    }
}

// the transformation matrix for the astronomical body to be sent to the gpu 
// since we can't send over a matrix, need to send over the vectors that comprise it
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AstroBodyInstanceRaw {
    mat: [[f32; 4]; 4],
    tex: u32,
}

impl AstroBodyInstanceRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] =
        vertex_attr_array![5 => Float32x4, 6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Uint32];

    // get the vertex buffer layout for the AstroBody instance
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<AstroBodyInstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
            // for each vec4. We'll have to reassemble the mat4 in the shader.
            attributes: &Self::ATTRIBUTES,
        }
    }
}


pub struct PlanetSim {
    pub objects: Vec<AstroBody>,
    focused_index: Option<usize>,  
    gravitational_constant: f64,
    // scale down distances when rendering so that objects arent *really* far away
    pub render_scale: f64,
    // this should be a fraction of the shortest orbit (otherwise it will decohere)
    min_timestep: Duration,
}

impl PlanetSim {
    pub fn new(gravitational_constant: f64, render_scale: f64, min_timestep: Duration) -> PlanetSim {
        PlanetSim {
            objects: Vec::new(),
            focused_index: None,
            gravitational_constant,
            render_scale,
            min_timestep,
        }
    }

    pub fn add(&mut self, body: AstroBody) {
        self.objects.push(body);
    }

    // removes the object associated with label from the sim
    // if it was the focused object, then sets the focused object to the default parameter if possible
    // if the focused object is none, it isn't replaced by the default
    // if force is true, it focuses new 0th planet
    pub fn remove(&mut self, label: &str, default_focus: Option<&str>, force: bool) {
        let old_focus_label: Option<String> = {
            if let Some(obj) = self.get_focused() {
                Some(obj.label.clone())
            } else {
                None
            }
        };
        
        // remove the associated label 
        for i in 0..self.len() {
            if self.objects[i].label == label {
                self.objects.remove(i);
                break;
            }
        }
        
        // set the new focus after removing to keep the indices consistent
        if let Some(old_focus_label) = old_focus_label {
            if old_focus_label == label {
                if !self.set_focused(default_focus) && force {
                    let forced_label = String::from(&self.objects[0].label);
                    self.set_focused(Some(&forced_label));
                } 
            } else {
                self.set_focused(Some(&old_focus_label));
            }
        }   
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn get_focused(&self) -> Option<&AstroBody> {
        match self.focused_index {
            // this branch should never return None, want to panic if that is the case as it indicates some error occurred
            Some(index) => Some(self.objects.get(index).expect(&format!("Focused object index {} was invalid", index))),
            None => None,
        }
    }

    pub fn get_mut_focused(&mut self) -> Option<&mut AstroBody> {
        match self.focused_index {
            // this branch should never return None, want to panic if that is the case as it indicates some error occurred
            Some(index) => Some(self.objects.get_mut(index).expect(&format!("Focused object index {} was invalid", index))),
            None => None,
        }
    }

    // returns whether or not it failed to set it, which is only if the str isnt a valid label
    pub fn set_focused(&mut self, label: Option<&str>) -> bool {
        match label {
            Some(new_label) => {
                for i in 0..self.len() {
                    if self.objects[i].label == new_label {
                        self.focused_index = Some(i);
                        return true;
                    } 
                }
                self.focused_index = None;
                return false;
            },
            None => self.focused_index = None,
        }
        return true;
    }

    // the timestep is time between updating in the simulation (which could be anywhere from milliseconds to years), not necessarily time between frames
    pub fn update(&mut self, timestep: Duration) {
        let mut time_remaining = timestep;
        
        // this will run timestep / min_timestep times, but will progress the sim by exactly timestep
        loop {
            let timestep = min(time_remaining, self.min_timestep);
    
            // update velocity
            for i in 0..self.objects.len() {
                // the body we want to update the velocity of
                let target = &self.objects[i];
                let mut new_velocity = target.velocity;
                
                for j in 0..self.objects.len() {
                    if i == j { continue; }
                    let obj = &self.objects[j];

                    // direction of the acceleration vector
                    // this points from target to obj
                    let acc_dir = (obj.position - target.position).normalize();
                    // distance squared between the two astronomical bodies
                    // since collisions aren't simulated, assume that the distance is always at least the two obj's radii apart
                    let mut dist_sq = target.position.distance2(obj.position);
                    let min_dist = target.radius + obj.radius; 
                    if dist_sq < min_dist * min_dist {
                        dist_sq = min_dist * min_dist;
                    }
                    // acceleration as calculated using the gravitational formula between two bodies
                    let acc = (self.gravitational_constant * obj.mass / dist_sq) * acc_dir;
                    
                    // add this portion of the acceleration to the velocity
                    new_velocity += acc * timestep.as_secs_f64(); 
                }

                // update the target's velocity to this new velocity after the given timestep
                let target = &mut self.objects[i];
                target.velocity = new_velocity;
            }

            // update position
            for body in &mut self.objects {
                body.position += body.velocity * timestep.as_secs_f64();
            }
            if let Some(obj) = self.get_focused() {
                let base_pos = obj.position;
                for body in &mut self.objects {
                    body.position -= base_pos;
                }
            }

            // TODO: update rotation 

            if let Some(t) = time_remaining.checked_sub(self.min_timestep) {
                time_remaining = t;
            } else {
                break;
            }
        }
    }

    // convert the PlanetSim and bodies into a bunch of instances to be rendered
    pub fn instance_data(&self) -> Vec<AstroBodyInstanceRaw> {
        self.objects.iter().map(|obj| obj.to_raw_instance(self.render_scale)).collect()
    }
}

impl PlanetSim {
    pub fn from_real_data() -> Self {
        let data = include_str!("../astronomical_body_data.csv");
        let mut rdr = ReaderBuilder::new()
            .delimiter(b';')
            .from_reader(data.as_bytes());

        let gravitational_constant = 0.000000000066743;
        // this is the result of adjusting the gravitational equation for distance in km and mass in 10^24 kg
        let scaled_gravitational_constant = gravitational_constant * 1_000_000_000_000_000.0;
        // render scale of 1_000_000 makes Earth on the scale of around ~6 units
        let render_scale = 1_000_000.0;
        // this is about 100th the time it should take the moon to orbit Earth, which is probably a good enough update
        let min_timestep = Duration::from_secs(27 * 24 * 60 * 60) / 100;
        let mut planet_sim = PlanetSim::new(scaled_gravitational_constant, render_scale, min_timestep);

        while let Some(result) = rdr.records().next() {
            let record = result.expect("Couldn't parse astronomical data csv record");
            
            let label = String::from(record.get(0).unwrap());
            let mass = record.get(1).unwrap().parse::<f64>().unwrap();
            let radius = record.get(2).unwrap().parse::<f64>().unwrap();
            let pos_x = record.get(3).unwrap().parse::<f64>().unwrap();
            let pos_y = record.get(4).unwrap().parse::<f64>().unwrap();
            let pos_z = record.get(5).unwrap().parse::<f64>().unwrap();
            let vel_x = record.get(6).unwrap().parse::<f64>().unwrap();
            let vel_y = record.get(7).unwrap().parse::<f64>().unwrap();
            let vel_z = record.get(8).unwrap().parse::<f64>().unwrap();
            let texture_index = record.get(9).unwrap().parse::<u32>().unwrap();
            let color_r = record.get(10).unwrap().parse::<f32>().unwrap();
            let color_g = record.get(11).unwrap().parse::<f32>().unwrap();
            let color_b = record.get(12).unwrap().parse::<f32>().unwrap();
            let color_a = record.get(13).unwrap().parse::<f32>().unwrap();

            let obj = AstroBody {
                label,
                texture_index,
                color: [color_r, color_g, color_b, color_a],
                mass,
                radius,
                position: Vector3::new(pos_x, pos_z, pos_y),
                velocity: Vector3::new(vel_x, vel_z, vel_y),
                rotation: Quaternion::zero(),
                axis_of_rotation: Vector3::unit_y(),
                angular_velocity: 0.0,
            };

            // println!("{:?}", obj);
            planet_sim.add(obj);
        }
        planet_sim.set_focused(Some("Sun"));

        planet_sim
    }
}