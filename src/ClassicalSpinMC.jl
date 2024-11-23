module ClassicalSpinMC

include("interaction_matrix.jl")

include("unit_cell.jl")
export UnitCell, addBasisSite!, addBilinear!, addCubic!, addQuartic!, addZeemanCoupling!, addOnSite!

include("reciprocal.jl")
export reciprocal, get_allowed_wavevectors, get_k_path, get_k_plane

include("lattice.jl")
export Lattice, set_spin!, random_spin_orientation

include("observables.jl")
export get_magnetization

include("helper.jl")

include("hdf5.jl")
export overwrite_keys!, write_MC_checkpoint, create_params_file, read_lattice, read_spin_configuration!

include("metropolis.jl")
export Metropolis, MetropolisAdaptive, MetropolisConstraint, MetropolisConstraintAdaptive

include("monte_carlo.jl")
export MonteCarlo, simulated_annealing!, deterministic_updates!, parallel_tempering!

include("hamiltonian.jl")
export total_energy, energy_density, get_local_field

include("bravais.jl")
export Triangular, Square, Honeycomb, FCC, Pyrochlore, BreathingPyrochlore

include("spin_correlations.jl")
export compute_equal_time_correlations, runEqualTimeStructureFactor!, compute_equal_time_structure_factor

end 
