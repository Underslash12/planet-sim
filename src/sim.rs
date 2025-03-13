// sim.rs

use cgmath::{InnerSpace, MetricSpace, Zero};
use web_time::{Instant, Duration};
use winit::dpi::Position;
use wgpu::vertex_attr_array;


pub struct AstroBody {
    mass: f64,
    radius: f64,
    position: cgmath::Vector3<f64>,
    velocity: cgmath::Vector3<f64>,
    rotation: cgmath::Quaternion<f32>,
    axis_of_rotation: cgmath::Vector3<f64>,
    angular_velocity: f64, 
}

impl AstroBody {
    pub fn new(mass: f64, radius: f64, position: cgmath::Vector3<f64>, velocity: cgmath::Vector3<f64>, 
        rotation: cgmath::Quaternion<f32>, axis_of_rotation: cgmath::Vector3<f64>, angular_velocity: f64
    ) -> AstroBody {
        AstroBody {
            mass,
            radius,
            position,
            velocity,
            rotation,
            axis_of_rotation,
            angular_velocity,
        }
    } 

    fn to_raw_instance(&self) -> AstroBodyInstanceRaw {
        let low_precision_position: cgmath::Vector3<f32> = cgmath::Vector3::new(self.position.x as f32, self.position.y as f32, self.position.z as f32);
        AstroBodyInstanceRaw {
            mat: (cgmath::Matrix4::from_translation(low_precision_position)).into(),
        }
    }
}

// the transformation matrix for the astronomical body to be sent to the gpu 
// since we can't send over a matrix, need to send over the vectors that comprise it
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AstroBodyInstanceRaw {
    mat: [[f32; 4]; 4],
}

impl AstroBodyInstanceRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
        vertex_attr_array![5 => Float32x4, 6 => Float32x4, 7 => Float32x4, 8 => Float32x4];

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
    objects: Vec<AstroBody>,
    gravitational_constant: f64,
}

impl PlanetSim {
    pub fn new(gravitational_constant: f64) -> PlanetSim {
        PlanetSim {
            objects: Vec::new(),
            gravitational_constant,
        }
    }

    pub fn add(&mut self, body: AstroBody) {
        self.objects.push(body);
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    // the timestep is time between updating in the simulation (which could be anywhere from milliseconds to years), not necessarily time between frames
    pub fn update(&mut self, timestep: Duration) {
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

        // TODO: update rotation 
    }

    // convert the PlanetSim and bodies into a bunch of instances to be rendered
    pub fn instance_data(&self) -> Vec<AstroBodyInstanceRaw> {
        self.objects.iter().map(AstroBody::to_raw_instance).collect()
    }
}


pub fn test_sim() {
    use std::thread;

    let mut ps = PlanetSim::new(6.67408 / 100_000_000_000.0);
    
    ps.add(AstroBody{
        mass: 100.0,
        radius: 1.0,
        position: cgmath::Vector3::new(-1.0, -1.0, 0.0),
        velocity: cgmath::Vector3::zero(),
        rotation: cgmath::Quaternion::zero(),
        axis_of_rotation: cgmath::Vector3::unit_z(),
        angular_velocity: 0.0,
    });

    ps.add(AstroBody{
        mass: 100.0,
        radius: 1.0,
        position: cgmath::Vector3::new(1.0, 1.0, 0.0),
        velocity: cgmath::Vector3::zero(),
        rotation: cgmath::Quaternion::zero(),
        axis_of_rotation: cgmath::Vector3::unit_z(),
        angular_velocity: 0.0,
    });

    let timestep = Duration::from_secs_f64(1.0 / 10.0);
    loop {
        ps.update(timestep);

        // for i in 0..ps.objects.len() {
        //     println!("Obj[{}]: {:?}", i, &ps.objects[i].position);
        // }
        // println!();
        
        let mut energy: f64 = 0.0;
        // compute potential energy
        for i in 0..ps.objects.len() {
            for j in 0..ps.objects.len() {
                if i == j { continue; }

                let dist = ps.objects[i].position.distance(ps.objects[j].position);
                let potential = -ps.gravitational_constant * ps.objects[i].mass * ps.objects[j].mass / dist;

                energy += potential;
            }
        }
        // compute kinetic energy
        for obj in &ps.objects {
            let speed_sq = obj.velocity.magnitude2();
            let kinetic = 0.5 * obj.mass * speed_sq;
            energy += kinetic;
        }

        println!("{}", energy);

        thread::sleep(timestep);
    }
}