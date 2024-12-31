import numpy as np
import openmm.app as app
import openmm as mm
import openmm.unit as unit
import scipy.constants as c
import os

import random

from .system_building import get_mpipi_system  # <--- now referencing get_mpipi_system
from .constants import PLATFORM, PROPERTIES

def calculate_target_box_vectors(chain_info, 
                                 long_side_scale_factor=6, 
                                 target_density=0.1*unit.gram/unit.centimeter**3):
    """
    Calculate the initial target box vectors for the system based on the total mass of the chains
    and a desired target density.

    Args:
        chain_info (dict): 
            A dictionary with keys = chain objects and values = number of copies to place.
            Each chain object must have:
              - chain_mass (in daltons)
              - min_rg_coords (initial coordinates)
        long_side_scale_factor (float, optional): 
            Scaling factor by which the box's first dimension is multiplied relative to the others.
        target_density (Quantity, optional): 
            Target density as an OpenMM `Quantity` (default 0.1 g/cm^3).

    Returns:
        Quantity: 
            A 3x3 array (in nm) of box vectors suitable for periodic boundary conditions.
    """
    total_mass_g = sum([chain.chain_mass.value_in_unit(unit.dalton)/c.Avogadro*unit.gram * n_copies 
                        for chain, n_copies in chain_info.items()], 0*unit.gram)
    
    target_volume = total_mass_g / target_density
    short_side_length = (target_volume / long_side_scale_factor) ** (1/3)
    short_side_length = short_side_length.in_units_of(unit.nanometer) / unit.nanometer

    return short_side_length * np.array([[long_side_scale_factor, 0, 0], 
                                         [0, 1, 0], 
                                         [0, 0, 1]]) * unit.nanometer

def build_model(chain_info, target_box_vectors):
    """
    Build a Modeller object with the given chain information and initial box vectors.

    The function arranges chains on a 3D grid to avoid overlaps as much as possible. 
    If needed, it scales the first dimension of the box to fit all chains.

    Args:
        chain_info (dict): 
            A dictionary with keys = chain objects and values = number of copies to place.
            Each chain object must have:
              - topology (OpenMM Topology)
              - min_rg_coords (initial coords)
              - max_rg (maximum radius of gyration)
              - min_rg (minimum radius of gyration)
        target_box_vectors (Quantity): 
            A 3x3 array of box vectors (in nm).

    Returns:
        Modeller: 
            A Modeller object containing the combined system of all chains, 
            set with appropriately extended periodic box vectors.
    """
    print('Initializing the model...', flush=True)

    first_chain = list(chain_info.keys())[0]
    model = app.Modeller(first_chain.topology, first_chain.min_rg_coords)

    max_max_rg = max(chain.max_rg for chain in chain_info.keys())

    box_vectors = target_box_vectors.value_in_unit(unit.nanometer)
    short_side_length = box_vectors[1][1]

    if short_side_length < 2.25 * max_max_rg:
        print(f"Warning! Short side length of {short_side_length:.2f} nm may be too small. "
              f"It should be at least 2.25 times the largest maximum Rg value ({max_max_rg:.2f} nm) in the system.")

    max_min_rg = max(chain.min_rg for chain in chain_info.keys())
    offset = 3.0 * max_min_rg

    max_copies_shortside = int(box_vectors[1][1] / offset)
    max_copies_longside = int(box_vectors[0][0] / offset)

    total_chains = sum(chain_info.values())
    larger_box_ratio = int(total_chains / (max_copies_longside * max_copies_shortside**2)) + 1

    assert larger_box_ratio * max_copies_longside * max_copies_shortside**2 >= total_chains

    extended_box_vectors = box_vectors.copy()
    extended_box_vectors[0][0] = larger_box_ratio * max_copies_longside * offset

    grid_positions = []
    for i in range(larger_box_ratio * max_copies_longside):
        for j in range(max_copies_shortside):
            for k in range(max_copies_shortside):
                position = np.array([i, j, k]) * offset
                grid_positions.append(position)

    chains_to_add = []
    for chain, n_copies in chain_info.items():
        # For the first chain, we add one less copy since it is already in the modeller
        if chain == first_chain:
            chains_to_add.extend([chain] * (n_copies - 1))
        else:
            chains_to_add.extend([chain] * n_copies)
    
    random.shuffle(chains_to_add)

    grid_index = 1  # Start from grid index 1 because 0 is for first_chain's original position
    for chain in chains_to_add:
        position = grid_positions[grid_index]
        model.add(chain.topology, (chain.min_rg_coords + position)*unit.nanometer)
        grid_index = (grid_index + 1) % len(grid_positions)

    model.topology.setPeriodicBoxVectors(extended_box_vectors*unit.nanometer)
    print('The model is built.', flush=True)
    print('The topology of the model:', model.topology)
    app.PDBFile.writeFile(model.topology, model.positions, open('./equi_model.pdb', 'w'))

    return model

def equilibrate_slab(model, 
                     target_box_vectors, 
                     chain_info, 
                     T=280*unit.kelvin, 
                     csx=150, 
                     pulling_time=20*unit.nanosecond, 
                     equi_time=400*unit.nanosecond):
    """
    Perform a two-stage equilibration:
    
    1) Pull all chains toward the center of the box to remove large voids (a "slab" pulling).
    2) Remove the pulling force, set the final box vectors, and equilibrate further.

    Args:
        model (Modeller): 
            A Modeller object with the combined system and initial positions.
        target_box_vectors (Quantity): 
            A 3x3 array of box vectors (in nm).
        chain_info (dict): 
            A dictionary with keys = chain objects and values = number of copies. 
            Used to retrieve `globular_indices`.
        T (Quantity, optional): 
            Temperature to run the simulation, default 280 K.
        csx (float, optional): 
            Ionic strength in mM, used to compute Debye length internally.
        pulling_time (Quantity, optional): 
            Duration of the pulling simulation (default 20 ns).
        equi_time (Quantity, optional): 
            Duration of the equilibration simulation after pulling (default 400 ns).
    """
    print('Setting up integrator...', flush=True)
    integrator = mm.LangevinMiddleIntegrator(T, 0.01/unit.picosecond, 10*unit.femtosecond)

    # Prepare the globular indices dictionary
    globular_indices_dict = {
        chain.chain_id: chain.globular_indices for chain in chain_info.keys()
    }

    print('Building mpipi system...', flush=True)
    # Calls your second-code function get_mpipi_system
    system = get_mpipi_system(
        np.array(model.positions), 
        model.topology, 
        globular_indices_dict, 
        T.value_in_unit(unit.kelvin), 
        csx, 
        CM_remover=False, 
        periodic=True
    )

    # Create a gentle pulling force to drive everything toward the box center
    pulling_force = mm.CustomExternalForce(
        'k*periodicdistance(x, y, z, x0, y0, z0)^2'
    )
    midpoint_x = 0.5*target_box_vectors[0][0]
    midpoint_y = 0.5*target_box_vectors[1][1]
    midpoint_z = 0.5*target_box_vectors[2][2]
    
    pulling_force.addGlobalParameter('x0', midpoint_x)
    pulling_force.addGlobalParameter('y0', midpoint_y)
    pulling_force.addGlobalParameter('z0', midpoint_z)
    pulling_force.addPerParticleParameter('k')

    # Add a small spring constant for each atom
    for atom in model.topology.atoms():
        pulling_force.addParticle(atom.index, [0.001])
    system.addForce(pulling_force)
    print('Pulling force is set.', flush=True)

    # Setting up the Simulation
    print('Initializing simulation...', flush=True)
    simulation = app.Simulation(
        model.topology, 
        system, 
        integrator, 
        platform=PLATFORM, 
        platformProperties=PROPERTIES
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setPeriodicBoxVectors(*model.topology.getPeriodicBoxVectors())

    # Minimize
    simulation.minimizeEnergy()
    print('Energy minimized.', flush=True)

    # Pulling stage
    simulation.reporters.append(
        app.StateDataReporter(
            './output_pulling.dat', 10000, 
            step=True, potentialEnergy=True, 
            temperature=True, density=True, elapsedTime=True
        )
    )
    simulation.reporters.append(app.XTCReporter('./traj_pulling.xtc', 10000))
    print('Beginning pulling...', flush=True)
    simulation.step(int(pulling_time/(10*unit.femtosecond)))
    print('Pulling complete.', flush=True)

    # Remove pulling force
    system.removeForce(system.getNumForces()-1)
    print('Pulling force removed.', flush=True)

    # Update box vectors
    simulation.context.setPeriodicBoxVectors(*target_box_vectors)
    print('Box vectors updated.', flush=True)
    
    # Re-initialize the context so that box changes take effect
    state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    simulation.context.reinitialize()
    simulation.context.setState(state)
    print('Context reinitialized.', flush=True)

    # Minimize again
    model.topology.setPeriodicBoxVectors(target_box_vectors)
    simulation.minimizeEnergy()

    # Clear old reporters
    simulation.reporters = []

    # Equilibration stage
    simulation.reporters.append(
        app.StateDataReporter(
            './output_equi.dat', 10000, 
            step=True, potentialEnergy=True, 
            temperature=True, elapsedTime=True
        )
    )
    simulation.reporters.append(app.XTCReporter('./traj_equi.xtc', 50000))
    print('Beginning equilibration...', flush=True)
    simulation.step(int(equi_time/(10*unit.femtosecond)))
    
    # Final state
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    app.PDBFile.writeFile(model.topology, positions, open('./equi_model.pdb', 'w'))
    simulation.saveState('equi_state.xml')
    print('Equilibration complete.', flush=True)

def build_and_equilibrate_model(chain_info, 
                                long_side_scale_factor=6, 
                                target_density=0.1*unit.gram/unit.centimeter**3, 
                                T=280*unit.kelvin, 
                                csx=150,
                                pulling_time=20*unit.nanosecond, 
                                equi_time=1000*unit.nanosecond):
    """
    Wrapper function to:
      1) Relax each chain monomer in isolation.
      2) Calculate and build the combined model with the desired box size.
      3) Perform slab-pulling followed by a full equilibration.

    Args:
        chain_info (dict):
            A dictionary with keys = chain objects and values = number of copies. 
            Each chain object must have:
              - get_compact_model(simulation_time): a method to relax the chain individually
              - chain_mass, min_rg_coords, max_rg, min_rg
        long_side_scale_factor (float, optional):
            Factor by which the first dimension of the box is scaled relative to the other two.
        target_density (Quantity, optional):
            Target density in g/cm^3.
        T (Quantity, optional):
            Simulation temperature.
        csx (float, optional):
            Ionic strength in mM (for Debye length calculations).
        pulling_time (Quantity, optional):
            Duration of the slab-pulling simulation.
        equi_time (Quantity, optional):
            Duration of the final equilibration simulation.
    """
    
    print('Relaxing monomers...')
    for chain in chain_info.keys():
        chain.get_compact_model(simulation_time=10*unit.nanosecond)
    
    # Build combined model
    target_box_vectors = calculate_target_box_vectors(chain_info, 
                                                      long_side_scale_factor, 
                                                      target_density)
    model = build_model(chain_info, target_box_vectors)

    # Equilibrate
    equilibrate_slab(model, target_box_vectors, chain_info, 
                     T=T, csx=csx, 
                     pulling_time=pulling_time, 
                     equi_time=equi_time)
