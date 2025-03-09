// main.rs

use planet_sim::run;

fn main() {
    pollster::block_on(run());
}