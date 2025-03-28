from OpenMpipi import * # import everything from OpenMpipi
# also includes importing openmm as mm, openmm.app as app and openmm.unit as unit
import numpy as np

# setting some parameters

n_copies = 200 # copies of protein to add
T = 300 # temperature in K
csx = 150 # [NaCl] in mM
final_box_length = 100 # dimension of the final cubic box in nm
npt_steps = int(1e6) # steps to run for the compression phase
droplet_steps = int(1e7) # steps to run for the equilibration of the droplet in larger cubic box

seq = 'GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF'
wt_a1 = IDP('wtA1', seq) # init the IDP object with the specified sequence
#wt_a1.get_compact_model() # this method runs a short single chain simulation and saves the most compact frame, useful for building models
topology = wt_a1.topology
positions = wt_a1.initial_coords #wt_a1.min_rg_coords

model = app.Modeller(topology, positions) # init the Modeller object, to combine Topology objects and coordinates

# find a suitable cubic number to add the required number of copies
total_to_add = n_copies 
cubes = [(i, i**3) for i in range(4, 11)]
for item in cubes:
    i, cubed = item
    if total_to_add < cubed:
        n_edge = i
        break

# preparing a grid of positions to place molecules on
print(n_edge)
grid_offsets = []
for i in range(n_edge):
    for j in range(n_edge):
        for k in range(n_edge):
            if [i, j, k] != [0, 0, 0]:
                grid_offsets.append([i, j, k])

# shuffling the grid is useful for reducing equilibration time in multi-component mixtures
random.shuffle(grid_offsets)

for i in range(total_to_add):
    offset = grid_offsets.pop()
    coords = np.array(offset) * 15.0 + positions # add fixed offset to the initial coords
    model.add(topology, coords * unit.nanometer) # then add the next copy to the Modeller object


# set the box vectors and save a frame of the original model
positions = np.array(model.positions.value_in_unit(unit.nanometer))
min_coords, max_coords = np.min(positions, axis=0), np.max(positions, axis=0)
box_size = np.max(max_coords - min_coords)
box_vectors = np.array([[box_size, 0, 0], [0, box_size, 0], [0, 0, box_size]])
model.topology.setPeriodicBoxVectors(box_vectors)
app.PDBFile.writeFile(model.topology, model.positions, open('./initial_model.pdb', 'w'))

# now we set up the System
# important comment: all OpenMpipi functions require coords as numpy arrays and without OpenMM units, hence using model.positions would throw
# an error here (model.positions would return a list of OpenMM Vec3 objects with units)
system = get_mpipi_system(positions, model.topology, {'wtA1': []}, T, csx, CM_remover=True, periodic=True)

# adding a barostat to the system to compress it into a dense cube
barostat = mm.MonteCarloBarostat(0.50*unit.atmosphere, T)
system.addForce(barostat)

# with the System ready, we can now prepare the Simulation object
integrator = mm.LangevinMiddleIntegrator(T, 0.01/unit.picosecond, 10*unit.femtosecond)
simulation = app.Simulation(model.topology, system, integrator, mm.Platform.getPlatformByName('CUDA'), {'Precision': 'Mixed'})

# set positions and box vectors in the Context, minimize
simulation.context.setPositions(positions)
simulation.context.setPeriodicBoxVectors(*box_vectors)
simulation.minimizeEnergy()

# add an XTC reporter and run the NpT steps
simulation.reporters.append(app.XTCReporter('npt.xtc', 10000))
simulation.step(npt_steps)

# save the state because we will need the positions and velocities
state = simulation.context.getState(getPositions=True, getVelocities=True)

# now remove the MonteCarloBarostat force from the System, update the box vectors and reinitialize the Context
for idx in range(system.getNumForces()):
    force = system.getForce(idx)
    if isinstance(force, mm.MonteCarloBarostat):
        system.removeForce(idx)
        break

simulation.context.reinitialize() # ensure the context is reinitialized after removing the barostat

# update box vectors to a cube with side length final_box_length
new_box_vectors = np.eye(3) * final_box_length  # a 3x3 identity matrix scaled by final_box_length
simulation.context.setPeriodicBoxVectors(*new_box_vectors)

# get current positions (centered on origin) and add a translation to center the molecules in the new box for convenience
positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
translation = np.array([final_box_length/2, final_box_length/2, final_box_length/2])
new_positions = positions + translation

# update positions in the simulation context with the new coordinates
simulation.context.setPositions(new_positions * unit.nanometer)

# continue the simulation in the NVT ensemble (droplet equilibration)
simulation.reporters = [app.XTCReporter('droplet.xtc', 10000)] # replace reporter with a new one for the droplet phase
simulation.step(droplet_steps)

# save the final model
model.topology.setPeriodicBoxVectors(new_box_vectors)
positions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(model.topology, positions, open('./final_model.pdb', 'w'))