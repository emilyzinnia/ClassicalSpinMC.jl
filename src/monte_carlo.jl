#monte_carlo.jl

using MPI
using Dates
using LinearAlgebra

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
                    overwrite::Bool=true, sigma0::Real=60)::MonteCarlo
    
    # trailing backslash included in outpath
    mc = MonteCarlo()
    mc.T = T 
    mc.observables = Observables()
    mc.lattice = deepcopy(lattice)
    mc.parameters = MCParamsBuffer(parameters)
    mc.lambda = 0.0
    mc.weight = weight
    mc.constraint = constraint 
    mc.sigma = sigma0
    mc.sigma0 = sigma0

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
Perform overrelaxation sweep. Reflects spins on each site along local field direction. 

Modifies "spin" matrix in Lattice object. 
"""
function overrelaxation!(lattice::Lattice)
    for site=1:lattice.size
        si = get_spin(lattice.spins, site)
        H = get_local_field(lattice, site)
        # if no local field, keep spin as is
        if H == (0.0, 0.0, 0.0)
            continue 
        end
        proj = 2.0 * dot(si, H) / (H[1]^2 + H[2]^2 + H[3]^2)
        newspin = (-si[1] + proj*H[1], -si[2]+proj*H[2], -si[3]+proj*H[3] ) 
        set_spin!(lattice.spins, newspin, site)
    end
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
function simulated_annealing!(mc::MonteCarlo, schedule::Function, T0::Float64=1.0; 
                              alg::FunctionWrapper{Float64, Tuple{MonteCarlo,Float64}}=Metropolis())
    T = T0
    time = 1
    out = length(mc.outpath) > 0

    accept_total = mc.parameters.t_thermalization*mc.lattice.size / mc.parameters.overrelaxation_rate

    while T > mc.T
        t = 1
        R = 0.0
        mc.sigma = mc.sigma0
        while t < mc.parameters.t_thermalization 
            overrelaxation!(mc.lattice)
            if t % mc.parameters.overrelaxation_rate == 0
                R += alg(mc, T)
            end
            t += 1
        end
        println("Acceptance rate at T=$T: $(round(R/accept_total*100, digits=5)) % ")
        T =  schedule(time)
        time +=1
        if out
            write_MC_checkpoint(mc)  
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
        if field == (0.0, 0.0, 0.0)
            sweeps +=1
            continue 
        end
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
function parallel_tempering!(mc::MonteCarlo, saveIC::Vector{Int64}=Vector{Int64}[]; 
    alg::FunctionWrapper{Float64, Tuple{MonteCarlo,Float64}}=Metropolis())
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
        else
            @warn "MPI commSize of 1; no replica exchanges will occur!"
        end
    else
        @error "MPI not initialized!"
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

    sweep = 0
    while sweep < total_sweeps 

        # overrelaxation sweeps
        overrelaxation!(mc.lattice)

        if sweep % mc.parameters.overrelaxation_rate == 0
            # local metropolis updates 
            accepted_local += alg(mc, mc.T)
            E = total_energy(mc.lattice)

            # attempt exchange
            if enableMPI && (sweep % mc.parameters.swap_rate == 0)

                # exchange energies to the right for odd and left for even t
                if (sweep / mc.parameters.swap_rate) % 2 == 0
                    partner_rank = iseven(rank) ? rank + 1 : rank - 1 
                else
                    partner_rank = iseven(rank) ? rank - 1 : rank + 1 
                end
                
                if partner_rank >= 0 && partner_rank < commSize 
                    T_partner = temp[partner_rank+1]

                    if iseven(rank) # if even rank, send then receive
                        E_partner = MPISimpleSendRecv(E, partner_rank, comm)
                    else # if odd, receive then send 
                        E_partner = MPISimpleRecvSend(E, partner_rank, comm)
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
                    end
                end
            end
        end

        # take measurements
        if sweep >= mc.parameters.t_thermalization 
            # write IC after thermalization
            if sweep % mc.parameters.checkpoint_rate == 0 
                if out
                    write_MC_checkpoint(mc)
                end
                if IC
                    timestep = (sweep - mc.parameters.t_thermalization) ÷ mc.parameters.checkpoint_rate
                    if any(rank .== saveIC)
                        write_initial_configuration(string(path,"/IC_$rank/IC_$timestep.h5"), mc)
                    end
                end
            end 

            # update observables 
            if sweep % mc.parameters.probe_rate == 0
                M = get_magnetization(mc.lattice)
                update_observables!(mc, E,  M)
            end
        end
        
        #increment sweep 
        sweep += 1
        
        # output runtime statistics 
        if sweep % mc.parameters.report_interval == 0
            output_stats[1] = accepted_local
            output_stats[2] = exchange_rate
            print_runtime_statistics!(mc, sweep, output_stats, enableMPI)
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
