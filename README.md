# AATPTPT
An Attempt To Parallelize The Powder Toy. This project will have OpenCL acceleration for falling-sand simulation, heat diffusion, pressure solving and some other parts of The Powder Toy and support multiple GPUs, CPUs and accelerators.

## Parallel algorithm is simple

### kernel 1
- for each pixel or cell that can store a single sand particle, compute a random direction to move to imitate probabilistic state of movement to make it look like solid, liquid or gas
- sand can move to only an empty spot on closest neighbors so only from those available points a single target is chosen, written to a variable in cell

### kernel 2
- for each cell, find all neighbors that required to move to current (center) pixel
- only pick one source cell as available and save the direction to another variable in cell

### kernel 3
- get all neighbor variables (those need to move towards center and those can accept the center if empty)
- find the only 1 matching pair (only 1 neighbor needs to go to/from center)
- compute: if center(current) cell is empty, then it is accepting sand from a neighbor, make it "1"(this means sand for now)
- compute: if center cell has a sand particle, then it is moving sand to a neighbor, make it "0"
- also compute: if none found, it stays as it is

There is also another kernel to do ping-ponging buffers to not cause any race-condition on updates on variables. Because all cells need to work on the original data, not updated data. This eliminates any bias-based artifacts.

## Performance for 1600x900 cells

RTX 4070 can do ~20k updates per second. Ryzen 7900 has 1200 updates per second. Integrated-GPU of Ryzen 7900 has 500 updates per second.

Extra performance can be gained by using 1 bit per pixel if there is only sand type of particle. Currently each pixel uses 1 byte so its possible to identify 255 different particle types.

Discrete GPUs would not lose much performance by adding new particles because currently it is bottlenecked by kernel-launch latency (10s of microseconds) and memory/cache bandwidth (100s of GB/s in RTX4070)

## OpenCL

All parallelization is made through OpenCL and many hardware vendors support it. 


## New Ideas

### Pressure for simulating water could be a cellular-automata too

Cons: 
- it will have a latency of traveling through cells (20k updates per second is fast enough but low-end GPUs would need optimization)
- may not behave very realistic (may not reach equal heights in two tubes connected)
- it requires another byte per cell at least (256 different pressure levels)
- requires extra thinking about all possible states of pressures around a sand particle before coding
Pros: 
- embarrassingly parallel, fully scalable from GT1030 to RTX4090 just like another sand simulation
- different phases of materials on macro-cosmos can be observed from simple rules defined in micro-cosmos (3x3 neighnoring cells interacting, trading values) without any hard-coded rules
- pressure can change the probability of a particle moving towards a direction so it can have an illusion of "movement speed" with a high-enough updates per second
- requires less coding (assuming thinking is complete) & bandwidth than a conventional pressure-solver (32bit float too bad for bandwidth)
