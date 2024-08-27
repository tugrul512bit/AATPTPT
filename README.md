# AATPTPT
An Attempt To Parallelize The Powder Toy. This project will have OpenCL acceleration for falling-sand simulation, heat diffusion, pressure solving and some other parts of The Powder Toy and support multiple GPUs, CPUs and accelerators.

Parallel algorithm is simple:

kernel 1:
- for each pixel or cell that can store a single sand particle, compute a random direction to move to imitate probabilistic state of movement to make it look like solid, liquid or gas
- sand can move to only an empty spot on closest neighbors so only from those available points a single target is chosen, written to a variable in cell

kernel 2:
- for each cell, find all neighbors that required to move to current (center) pixel
- only pick one source cell as available and save the direction to another variable in cell

kernel 3:
- get all neighbor variables (those need to move towards center and those can accept the center if empty)
- find the only 1 matching pair (only 1 neighbor needs to go to/from center)
- compute: if center(current) cell is empty, then it is accepting sand from a neighbor, make it "1"(this means sand for now)
- compute: if center cell has a sand particle, then it is moving sand to a neighbor, make it "0"
- also compute: if none found, it stays as it is

There is also another kernel to do ping-ponging buffers to not cause any race-condition on updates on variables. Because all cells need to work on the original data, not updated data. This eliminates any bias-based artifacts.
