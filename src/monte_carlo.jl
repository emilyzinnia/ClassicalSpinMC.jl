#monte_carlo.jl

using MPI
using Dates
using LinearAlgebra

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
    sweep::Int64   
    lambda::Float64  # total penalty imposed by constraints
    parameters::SimulationParameters
    observables::Observables
    lattice::Lattice 
    roundtripMarker::Float64
    weight::Float64
    constraint::Function      # lagrange multipliers for constraints 
    outpath::String # absolute path to hdf5 file 
    MonteCarlo() = new()
end

"""
Buffer for creating SimulationParameters from a Dict of values. 
"""
function MCParamsBuffer(dict::Dict{String,Int64})::SimulationParameters
    allowed_keys = ["t_thermalization", "t_deterministic","t_measurement", "probe_rate", "swap_rate", "overrelaxation_rate", "report_interval", "checkpoint_rate"]
    default_vals = [1, 1, 1, 1, 1, 10, 0, 0]
    ordered_vals = []
    for (i,key) in enumerate(allowed_keys)
        if !(key in keys(dict))
            dict[key] = default_vals[i]
        end
        push!(ordered_vals, dict[key])
    end

    for key in keys(dict)
        if !(key in allowed_keys)
            @warn "'$key' not a valid MC parameter; ignoring"
        end
    end
    return SimulationParameters(ordered_vals...)
end 

"""
Wrapper for creating a MonteCarlo object. 

# Arguments
- `T::Float64`: MC target temperature.
- `lattice::Lattice`: Lattice object. See Lattice documentation. 
- `parameters::Dict{String,Int64}`: dictionary containing MC parameters. Allowed keys include 
    - `t_thermalization::Integer`: # of MC thermalization sweeps.
    - `t_deterministic::Integer`: # of deterministic update sweeps. 
    - `t_measurement::Integer`: # of MC measurement sweeps.
    - `probe_rate::Integer`: take a measurement every probe_rate sweeps.
    - `swap_rate::Integer`: # attempt a replica exchange every swap_rate sweeps.
    - `overrelaxation_rate::Integer`: # of overrelaxation sweeps to perform for every Metropolis sweep. 
    - `report_interval::Integer`: print out exchange statistics every report_interval.
    - `checkpoint_rate::Integer`: rate at which a checkpoint is written, and if specified, 
    how often an IC is outputted.

# Keyword Arguments 
- `constraint::Function=x->0.0`: perform Metropolis sweep with specified constraint.
- `weight::Float64=0.0`: weight of the constraint. if weight is 0, use unconstrained Metropolis function. 
- `outpath::String=""`: path to directory to write .h5 files to with trailing backslash. 
if empty string, no files are written. 
- `outprefix::String="configuration"`: prefix of filename(s) for output. 
- `inparams::Dict{String,<:Any}=Dict{String,<:Any}()`: optional dictionary of simulation parameters for 
output in .params file. 
- `overwrite::Bool=true`: flag for overwriting existing files. 
"""
function MonteCarlo(T::Float64, lattice::Lattice, parameters::Dict{String,Int64}; 
                    constraint::Function=x->0.0, weight::Float64=0.0, 
                    outpath::String="", outprefix::String="configuration", 
                    inparams::Dict{String,<:Any}=Dict{String,Any}(),
                    overwrite::Bool=true)::MonteCarlo
    
    # trailing backslash included in outpath
    mc = MonteCarlo()
    mc.T = T 
    mc.sweep = 0
    mc.observables = Observables()
    mc.lattice = deepcopy(lattice)
    mc.parameters = MCParamsBuffer(parameters)
    mc.roundtripMarker = 1.0
    mc.lambda = 0.0
    mc.weight = weight
    mc.constraint = constraint 

    # initialize MPI parameters 
    if MPI.Initialized()
        comm = MPI.COMM_WORLD
        commSize = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
    else
        rank = 0
        commSize = 1
    end

    # check if configuration file exists. if not, create it 
    if length(outpath) > 0 
        filename=string(outprefix,"_",rank,".h5")
        rank == 0 && !isdir(string(outpath)) && mkdir( string(outpath) )
        mc.outpath = string(outpath,filename)
        paramsfile = string(outpath,outprefix,".h5.params")

        # create params file if doesn't exist (dumping metadata)
        if rank == 0 && !isfile(paramsfile) && overwrite 
            create_params_file(mc, paramsfile)
            if length(inparams) > 0
                write_attributes(paramsfile, inparams)
            end
        end

        # create hdf5 containing initial spin configuration on rank 
        if !isfile(mc.outpath) && overwrite
            println("Creating new file $filename for output on rank $rank")
            initialize_hdf5(mc, paramsfile)
        end
    else
        mc.outpath = outpath # outpath is empty string 
    end
    return mc
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

"""
Perform overrelaxation sweep. Reflects spins on each site along local field direction. 

Modifies "spin" matrix in Lattice object. 
"""
function overrelaxation!(lattice::Lattice)
    for site=1:lattice.size
        si = get_spin(lattice.spins, site)
        H = get_local_field(lattice, site)
        proj = 2.0 * dot(si, H) / (H[1]^2 + H[2]^2 + H[3]^2)
        newspin = (-si[1] + proj*H[1], -si[2]+proj*H[2], -si[3]+proj*H[3] ) 
        set_spin!(lattice.spins, newspin, site)
    end
end

"""
Perform Metropolis sweep with specified constraint. 

Returns a count of how many sweeps were accepted for statistics. 
"""
function metropolis_constraint!(mc::MonteCarlo, T::Float64)
    accept_rate = 0.0
    # perform local updates and sweep through lattice
    sweep = 0
    while sweep < mc.lattice.size
        point = rand(1:mc.lattice.size) # pick random index
        old_spin = get_spin(mc.lattice.spins, point) # store old spin 
        delta_E = calculate_energy_diff!(mc.lattice, point) 
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
function metropolis!(mc::MonteCarlo, T::Float64)
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

"""
Slow annealing from high temperature to desired temperature.

Performs `overrelaxation!` and `metropolis!` or `metropolis_constraint!` sweeps at temperature T, 
then lowers temperature according to annealing schedule. 

# Arguments 
- `mc::MonteCarlo`: MonteCarlo object containing target temperature mc.T. MC parameters to be specified:  
    - `t_thermalization`: thermalization time 
    - `overrelaxation_rate`: overrelaxation rate. by default, a metropolis step is performed 
    every 10 overrelaxation sweeps. 
- `schedule::Function`: annealing schedule function for lowering temperature. 
new T = schedule(time) where time is incremented by 1. 
- `T0::Float64=1.0`: initial high temperature.

"""
function simulated_annealing!(mc::MonteCarlo, schedule::Function, T0::Float64=1.0)
    T = T0
    time = 1
    out = length(mc.outpath) > 0

    # determine whether to use constrained metropolis or unconstrained
    if mc.weight != 0
        met = metropolis_constraint!
    else
        met = metropolis!
    end

    while T > mc.T
        t = 1
        while t < mc.parameters.t_thermalization
            overrelaxation!(mc.lattice)
            if t % mc.parameters.overrelaxation_rate == 0
                met(mc, T)
            end
            t += 1
        end
        T =  schedule(time)
        time +=1
        if out
            println("Lowering temperature to T=$T and writing checkpoint")
            write_MC_checkpoint(mc)  
        else
            println("Lowering temperature to T=$T")
        end
    end
end

"""
Align spins to local field. 

Usually used after annealing to low T(~1e-7) when acceptance rate is low. 

# Arguments 
- `mc::MonteCarlo`: MonteCarlo object. MC parameters to be specified:  
    - `t_deterministic`: # of deterministic update sweeps to perform 
"""
function deterministic_updates!(mc::MonteCarlo)
    sweeps = 1
    while sweeps < mc.parameters.t_deterministic
        point = rand(1:mc.lattice.size) # pick random index
        field = get_local_field(mc.lattice, point)
        set_spin!(mc.lattice.spins, .-field ./ norm(field) .* mc.lattice.S, point)
        sweeps +=1 
    end
end

"""
Performs parallel tempering algorithm. 

Configurations at various temperatures are launched in parallel using MPI. 
Overrelaxation sweeps and metropolis sweeps are performed for each temperature, 
and replica exchanges are attempted according to the `swap_rate`.

# Arguments
- `mc::MonteCarlo`: MonteCarlo object containing target temperature for each rank. MC parameters to be specified
when initializing MonteCarlo object:  
    - `t_thermalization`
    - `t_measurement`
    - `probe_rate`
    - `swap_rate`
    - `overrelaxation_rate`
    - `report_interval`
    - `checkpoint_rate`
- `saveIC::Vector{Int64}=Vector{Int64}[]`: vector containing which ranks to output thermalized 
configurations on every `checkpoint_rate` sweeps. can be used as initial configurations (IC) for Landau Lifshitz Gilbert calculations. 
"""
function parallel_tempering!(mc::MonteCarlo, saveIC::Vector{Int64}=Vector{Int64}[])
    # initialize MPI parameters 
    rank = 0
    commSize = 1
    enableMPI = false

    # write only if outpath specified during initialization of MC object 
    out = length(mc.outpath) > 0 

    # if MPI initialized, collect temperatures on each rank 
    if MPI.Initialized()
        comm = MPI.COMM_WORLD
        commSize = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        if commSize > 1
            # get temperature with rank 
            temp = zeros(commSize)
            temp[rank+1] = mc.T
            commSize > 1 && MPI.Allgather!(MPI.IN_PLACE, temp, 1, MPI.COMM_WORLD) # collect temps from all ranks
            T = temp[rank+1]
            enableMPI = true
        end
    end

    if mc.weight != 0 
        met = metropolis_constraint!
    else
        met = metropolis!
    end

    # generate initial spins and initialize energy  
    E = total_energy(mc.lattice)
    new_spins = similar(mc.lattice.spins)

    # initialize output MC statistics 
    accepted_local = 0
    exchange_rate = 0
    exchange_rate_prev = 0
    s_prev = 0
    local_prev = 0
    output_stats = [accepted_local, exchange_rate, exchange_rate_prev, s_prev, local_prev]
    accept_arr = [false]
    total_sweeps = mc.parameters.t_thermalization + mc.parameters.t_measurement

    IC = length(saveIC) != 0
    path = dirname(mc.outpath)
    if IC
        any(rank .== saveIC) && println("Initializing IC collection on rank $rank")
        any(rank .== saveIC) && !isdir(string(path, "/IC_$rank")) && mkdir( string(path, "/IC_$rank") )
    end

    # run parallel tempering
    rank == 0 && @printf("Running sweeps on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))

    while mc.sweep < total_sweeps 

        # overrelaxation sweeps
        overrelaxation!(mc.lattice)

        if mc.sweep % mc.parameters.overrelaxation_rate == 0
            # local metroppolis updates 
            accepted_local += met(mc, mc.T)
            E = total_energy(mc.lattice)

            # attempt exchange
            if enableMPI && (mc.sweep % mc.parameters.swap_rate == 0)

                # exchange energies to the right for odd and left for even t
                if (mc.sweep / mc.parameters.swap_rate) % 2 == 0
                    partner_rank = iseven(rank) ? rank + 1 : rank - 1 
                else
                    partner_rank = iseven(rank) ? rank - 1 : rank + 1 
                end
                
                if partner_rank >= 0 && partner_rank < commSize 
                    T_partner = temp[partner_rank+1]

                    if iseven(rank) # if even rank, send then receive
                        E_partner = MPISimpleSendRecv(E, partner_rank, comm)
                        roundtrip_partner = MPISimpleSendRecv(mc.roundtripMarker, partner_rank, comm)

                    else # if odd, receive then send 
                        E_partner = MPISimpleRecvSend(E, partner_rank, comm)
                        roundtrip_partner = MPISimpleRecvSend(mc.roundtripMarker, partner_rank, comm)
                    end

                    accept_arr[1] = false
                    if iseven(rank) # attempt the swap on the even rank 
                        delta_beta = (1/T_partner - 1/T)
                        delta_E = (E_partner-E)
                        accept_arr[1] = (rand() < min(1.0, exp(delta_beta*delta_E))) ? true : false
                        MPI.Send(accept_arr, partner_rank, 1, comm)
                    else
                        MPI.Recv!(accept_arr, partner_rank, 1, comm)
                    end

                    if accept_arr[1] # do the exchange 
                        exchange_rate += 1
                        if iseven(rank)
                            MPI.Sendrecv!(mc.lattice.spins, partner_rank, 0,
                                        new_spins, partner_rank, 0, comm)
                        else
                            MPI.Recv!(new_spins, partner_rank, 0, comm)
                            MPI.Send(mc.lattice.spins, partner_rank, 0, comm)
                        end
                        mc.lattice.spins = copy(new_spins)
                        E = E_partner
                        mc.roundtripMarker = roundtrip_partner
                    end
                end
                # for checking MC convergence 
                if rank == 0
                    mc.roundtripMarker = 0.0
                elseif rank == commSize - 1
                    mc.roundtripMarker = 1.0
                end 
            end
        end

        # take measurements
        if mc.sweep >= mc.parameters.t_thermalization 
            push!(mc.observables.roundtripMarker, mc.roundtripMarker) 
            
            # write IC after thermalization
            if mc.sweep % mc.parameters.checkpoint_rate == 0 
                if out
                    write_MC_checkpoint(mc)
                end
                if IC
                    timestep = (mc.sweep - mc.parameters.t_thermalization) ÷ mc.parameters.checkpoint_rate
                    if any(rank .== saveIC)
                        write_initial_configuration(string(path,"/IC_$rank/IC_$timestep.h5"), mc)
                    end
                end
            end 

            # update observables 
            if mc.sweep % mc.parameters.probe_rate == 0
                M = get_magnetization(mc.lattice)
                update_observables!(mc, E,  M)
            end
        end
        
        #increment sweep 
        mc.sweep += 1
        
        # output runtime statistics 
        if mc.sweep % mc.parameters.report_interval == 0
            output_stats[1] = accepted_local
            output_stats[2] = exchange_rate
            print_runtime_statistics!(mc, output_stats, enableMPI)
        end
    end

    if out
        # output observables 
        rank == 0 && @printf("Writing observables on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
        write_final_observables(mc)
    end

    rank == 0 && @printf("Simulation finished on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
    return 
end
