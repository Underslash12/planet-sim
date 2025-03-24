# planet-sim

planet-sim is a semi-realistic n-body planetary simulation that runs off of webgl and wasm, and is written in rust. I wrote this for class, so it isn't too polished, but the simulation itself is stable, and the main features are there, including camera controls, the n-body simulation itself, and the ability to add and remove planets, as well as changing their values while it is running. 

The browser version should work on Edge, Firefox, and Chrome, and is untested on Safari.    

## Features

n-body simulation: All the planets are simulated fairly accurately in that their starting orbits (taken from JPL's Horizons API) are stable. Since this is an n-body simulation, adding other planets, changing the position / velocity of planets, and changing their mass, will all affect the simulation results. Since this is a naive implementation, adding a significant number of planets will slow it down, but for reasonable amounts (under 50) it should be fine.

fast backend: Even with the naive implementation, the rust / wasm backend allows it to simulate anywhere from real time to 31 years of simulation time per real-life second. 

accurate values: As mentioned above, the planetary data position and velocity data was taken from JPL's Horizons API, so the starting point of the simulation is very accurate, and it seems like it stays accurate for a while. If you want to add in other real-life objects, the start date of the simulation is currently March 1st, 2025, with data taken from the vectors ephemeris database. Also, the y and z coordinates are swapped.  

## Controls

W - move forward\
S - move backward\
A - move left\
D - move right\
Space - move up\
Shift - move down\
Left Arrow - turn left\
Right Arrow - turn right\
Up Arrow - turn up\
Down Arrow - turn down\
Scroll up - increase movement speed\
Scroll down - decrease movement speed

The side panel has a slider to change the time scale, and it can run up to 31 years of simulation time per second of real time (though it may not be able to run that fast due to performance limitations). It also has input fields for the planet stats, but due to how I implemented the internals, the position is displayed as relative to the target body (which is itself) so its position is <0, 0, 0>. However, you can still add relative position to it, which does work, though visually they will be overwritten when saving the value. Changing the radius of the planet is purely visual, only mass affects the actual simulation. 

If you are ever unable to figure out where you are, look straight down, increase your movement speed to the max, and fly upwards. Unless all the planets have been ejected, eventually the dots will start to appear. Alternatively, the majority of the planets live roughly on the horizon, so they can be found there as well. 

## Building

To run in a native window, use `cargo run --release`. This does not have the side panel for controls, so all you can do is basically just fly around. 

To run a local instance of the server, install wasm-pack using `cargo install wasm-pack`, then run `wasm-pack build --target web` to compile the wasm to the `pkg` folder. Then run `server.py` using python 3 and open the site it links to.   

Of course, make sure rust is also installed.