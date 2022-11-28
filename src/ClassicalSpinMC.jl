module ClassicalSpinMC


include("interaction_matrix.jl")

include("unit_cell.jl")
export UnitCell, addBasisSite!, addInteraction!, addRingExchange!, addZeemanCoupling!

include("reciprocal.jl")
export get_allowed_wavevectors, get_k_path, get_k_plane

include("lattice.jl")
export lattice, set_spin!, random_spin_orientation

include("observables.jl")
export get_magnetization

include("hdf5.jl")
export overwrite_keys!, dump_attributes_hdf5!, initialize_hdf5, write_MC_checkpoint!

include("helper.jl")
include("monte_carlo.jl")
export SimulationParameters, MonteCarlo, simulated_annealing!, deterministic_updates!, parallel_tempering!

include("hamiltonian.jl")
export total_energy, energy_density, get_local_field

include("bravais.jl")
export Triangular, Square, Honeycomb, FCC, Pyrochlore, BreathingPyrochlore

include("molecular_dynamics.jl")
export compute_equal_time_correlations, runStaticStructureFactor!, runMolecularDynamics!, compute_static_structure_factor, compute_dynamic_structure_factor
end 
