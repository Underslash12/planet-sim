// main.rs

use planet_sim::run;
mod sim;

fn main() {
    pollster::block_on(run());
    // sim::test_sim();
    // cgmath::Matrix3::from_angle_x(cgmath::Rad(0.0)) * cgmath::Vector3::zero();
}