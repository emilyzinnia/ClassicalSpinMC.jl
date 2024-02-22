using Random, LinearAlgebra
using FunctionWrappers: FunctionWrapper

struct SimulationParameters
    t_thermalization::Int64 # total thermalization sweeps 
    t_deterministic::Int64 # total deterministic updates 
    t_measurement::Int64 # total measurement sweeps 
    probe_rate::Int64 # rate at which measurements are taken after thermalization
    swap_rate::Int64 # rate at which replica exchanges are attempted 
    overrelaxation_rate::Int64 # ratio of overrelaxation sweeps : metropolis sweeps
    report_interval::Int64 
    checkpoint_rate::Int64 # rate at which checkpoints are written as ICs for MD 
end

mutable struct MonteCarlo
    T::Float64    # temperature
    parameters::SimulationParameters
    lattice::Lattice 
    observables::Observables
    lambda::Float64  # total penalty imposed by constraints
    weight::Float64
    constraint::Function      # lagrange multipliers for constraints 
    outpath::String # absolute path to hdf5 file 
    sigma::Real
    sigma0::Real
    MonteCarlo() = new()
end

"""
Perform Metropolis sweep with specified constraint. 

Returns a count of how many sweeps were accepted for statistics. 
"""
function metropolis_constraint!(mc::MonteCarlo, T::Float64)::Float64
    accept_rate = 0.0
    # perform local updates and sweep through lattice
    sweep = 0

    while sweep < mc.lattice.size
        point = rand(1:mc.lattice.size) # pick random index
        old_spin = get_spin(mc.lattice.spins, point) # store old spin 
        delta_E = e_diff(mc.lattice, point) 
        c = mc.constraint(mc.lattice)
        new_lambda = mc.lambda - mc.weight * c

        accept = (delta_E-(new_lambda * c)) < 0 ? true : rand() < exp(-(delta_E-(new_lambda * c)) / T)
        if !accept
            set_spin!(mc.lattice.spins, old_spin, point) 
        else 
            accept_rate += 1 
            mc.lambda = new_lambda 
        end
        sweep += 1 
    end
    return accept_rate
end

"""
Perform Metropolis sweep. 

Returns a count of how many sweeps were accepted for statistics. 
"""
function metropolis!(mc::MonteCarlo, T::Float64)::Float64
    accept_rate = 0.0
    # perform local updates and sweep through lattice
    sweep = 0
    while sweep < mc.lattice.size
        point = rand(1:mc.lattice.size) # pick random index
        old_spin = get_spin(mc.lattice.spins, point) # store old spin 
        delta_E = calculate_energy_diff!(mc.lattice, point) 
        accept = (delta_E) < 0 ? true : rand() < exp(-(delta_E) / T)
        if !accept
            set_spin!(mc.lattice.spins, old_spin, point) 
        else 
            accept_rate += 1 
        end
        sweep += 1 
    end
    return accept_rate
end

function gaussian_move(S::Real, spin::NTuple{3,Float64}, sigma::Real=60)::NTuple{3,Float64}
    newspin = spin .+ (sigma .* random_spin_orientation(S))
    return newspin ./ norm(newspin) * S
end

"""
Calculates energy difference after choosing a random spin direction on site `point`.

Modifies "spin" matrix in Lattice object. 
"""
function calculate_energy_diff!(lattice::Lattice, point::Int64)::Float64
    E_old = energy(lattice, point)
    r = random_spin_orientation(lattice.S)
    set_spin!(lattice.spins, r, point) #flip spin direction at point
    E_new = energy(lattice, point)
    delta_E = E_new - E_old
    return delta_E
end

function calculate_energy_diff!(lattice::Lattice, point::Int64, sigma::Real)::Float64
    E_old = energy(lattice, point)
    r = gaussian_move(lattice.S, get_spin(lattice.spins, point), sigma)
    set_spin!(lattice.spins, r, point) 
    E_new = energy(lattice, point)
    delta_E = E_new - E_old
    return delta_E
end

# need to reset mc.sigma for every new temperature 
function metropolis_adaptive!(mc::MonteCarlo, T::Float64)::Float64
    accept_rate = 0.0
    # perform local updates and sweep through lattice
    sweep = 0
    while sweep < mc.lattice.size
        point = rand(1:mc.lattice.size) # pick random index
        old_spin = get_spin(mc.lattice.spins, point) # store old spin 
        delta_E = calculate_energy_diff!(mc.lattice, point, mc.sigma) 
        accept = (delta_E) < 0 ? true : rand() < exp(-(delta_E) / T)
        if !accept
            set_spin!(mc.lattice.spins, old_spin, point) 
        else 
            accept_rate += 1 
        end
        sweep += 1 
    end
    f = 0.5 / (1-accept_rate/sweep)
    # mc.sigma = accept_rate/sweep < 0.5 ? mc.sigma * f : mc.sigma0 # if acceptance rate > 50%, reset cone width to initial large value 
    mc.sigma *= f 
    return accept_rate
end

function metropolis_constraint_adaptive!(mc::MonteCarlo, T::Float64)::Float64
    accept_rate = 0.0
    # perform local updates and sweep through lattice
    sweep = 0

    while sweep < mc.lattice.size
        point = rand(1:mc.lattice.size) # pick random index
        old_spin = get_spin(mc.lattice.spins, point) # store old spin 
        delta_E = calculate_energy_diff!(mc.lattice, point, mc.sigma) 
        c = mc.constraint(mc.lattice)
        new_lambda = mc.lambda - mc.weight * c

        accept = (delta_E-(new_lambda * c)) < 0 ? true : rand() < exp(-(delta_E-(new_lambda * c)) / T)
        if !accept
            set_spin!(mc.lattice.spins, old_spin, point) 
        else 
            accept_rate += 1 
            mc.lambda = new_lambda 
        end
        sweep += 1 
    end
    f = 0.5 / (1-accept_rate/sweep)
    mc.sigma = accept_rate/sweep < 0.5 ? mc.sigma * f : mc.sigma0 # if acceptance rate > 50%, reset cone width to initial large value 
    return accept_rate
end

function Metropolis()::FunctionWrapper{Float64, Tuple{MonteCarlo,Float64}}
    return FunctionWrapper{Float64, Tuple{MonteCarlo,Float64}}(metropolis!)
end

function MetropolisAdaptive()::FunctionWrapper{Float64, Tuple{MonteCarlo,Float64}}
    return FunctionWrapper{Float64, Tuple{MonteCarlo,Float64}}(metropolis_adaptive!)
end

function MetropolisConstraint()::FunctionWrapper{Float64, Tuple{MonteCarlo,Float64}}
    return FunctionWrapper{Float64, Tuple{MonteCarlo,Float64}}(metropolis_constraint!)
end

function MetropolisConstraintAdaptive()::FunctionWrapper{Float64, Tuple{MonteCarlo,Float64}}
    return FunctionWrapper{Float64, Tuple{MonteCarlo,Float64}}(metropolis_constraint_adaptive!)
end