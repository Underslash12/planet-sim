// main.rs

use planet_sim::run;
mod sim;

fn main() {
    pollster::block_on(run());
    // sim::test_sim();
}