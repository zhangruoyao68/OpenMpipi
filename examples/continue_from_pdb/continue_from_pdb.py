# continuing an Mpipi simulation from a pdb file
from OpenMpipi import *

# set initial parameters
T = 300 # temperature in Kelvin
csx = 150 # salt concentration in mM
steps = int(1e7) # steps to run

# init IDP object
seq = 'GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF'
wt_A1 = IDP('wtA1', seq) # init the IDP object with the specified sequence

# get positions and Topology from pdb
pdb = app.PDBFile('initial_model.pdb')
positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
topology = pdb.getTopology()

# go through the chains in the pdb Topology and set their chain ID (though this is only actually important if you are simulating
# a protein with globular domains-- this is how the code knows which residues to include in the elastic networks
for chain in topology.chains():
  chain.id = wt_A1.chain_id

# set up the System
# important comment: all OpenMpipi functions require coords as numpy arrays and without OpenMM units, hence using model.positions would throw
# an error here (model.positions would return a list of OpenMM Vec3 objects with units)
system = get_mpipi_system(positions, topology, {'wtA1': []}, T, csx, CM_remover=True, periodic=True)

# with the System ready, we can now prepare the Simulation object
integrator = mm.LangevinMiddleIntegrator(T, 0.01/unit.picosecond, 10*unit.femtosecond)
simulation = app.Simulation(topology, system, integrator, mm.Platform.getPlatformByName('CUDA'), {'Precision': 'Mixed'})

# set positions and box vectors in the Context, minimize
simulation.context.setPositions(positions)
simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
simulation.minimizeEnergy()

# add reporters and run the simulation
simulation.reporters.append(app.XTCReporter('trajectory.xtc', 10000))
simulation.reporters.append(app.StateDataReporter('state.out', 10000, step=True, potentialEnergy=True, temperature=True, elapsedTime=True))
simulation.step(steps)
