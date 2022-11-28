#monte_carlo.jl

using MPI
using Dates
using LinearAlgebra

struct SimulationParameters
    t_thermalization::Int64 # total thermalization sweeps 
    t_measurement::Int64 # total measurement sweeps 
    probe_rate::Int64 # rate at which measurements are taken after thermalization
    swap_rate::Int64 # rate at which replica exchanges are attempted 
    OR::Int64 # ratio of overrelaxation sweeps : metropolis sweeps
    report_interval::Int64 
    checkpoint_rate::Int64 # rate at which checkpoints are written as ICs for MD 
end

mutable struct MonteCarlo
    T::Float64
    sweep::Int64

    input_parameters::Dict{String,Float64}
    parameters::SimulationParameters
    observables::Observables
    lattice::Lattice 
    roundtripMarker::Float64
    MonteCarlo() = new()
end

#MonteCarlo wrapper
function MonteCarlo(T::Float64, lattice::Lattice, parameters::SimulationParameters, inparams::Dict{String,Float})::MonteCarlo
    mc = MonteCarlo()
    mc.T = T 
    mc.sweep = 0
    mc.observables = Observables()
    mc.lattice = deepcopy(lattice)
    mc.parameters = deepcopy(parameters)
    mc.input_parameters = deepcopy(inparams)
    mc.roundtripMarker = 1.0
    return mc
end

# calculates energy difference. modifies the "spin" matrix
function calculate_energy_diff!(lattice::Lattice, point::Int64)::Float64
    E_old = energy(lattice, point)
    r = random_spin_orientation(lattice.S)
    set_spin!(lattice.spins, r, point) #flip spin direction at point
    E_new = energy(lattice, point)
    delta_E = E_new - E_old
    return delta_E
end

function overrelaxation!(lattice::Lattice)
    for site=1:lattice.size
        si = get_spin(lattice.spins, site)
        H = get_local_field(lattice, site)
        proj = 2.0 * dot(si, H) / (H[1]^2 + H[2]^2 + H[3]^2)
        newspin = (-si[1] + proj*H[1], -si[2]+proj*H[2], -si[3]+proj*H[3] ) 
        set_spin!(lattice.spins, newspin, site)
    end
end

function metropolis!(mc::MonteCarlo, T::Float64)
    accept_rate = 0.0
    # perform local updates and sweep through lattice
    sweep = 0
    while sweep < mc.lattice.size
        point = rand(1:mc.lattice.size) # pick random index
        old_spin = get_spin(mc.lattice.spins, point) # store old spin 
        delta_E = calculate_energy_diff!(mc.lattice, point) # calculate energy difference 
        accept = delta_E < 0 ? true : rand() < exp(-delta_E / T)
        if !accept
            set_spin!(mc.lattice.spins, old_spin, point) # if not accepted, revert back to old spin config
        else 
            accept_rate += 1 
        end
        sweep += 1 
    end
    return accept_rate
end

"""
Slow annealing from high temperature to desired temperature.
"""
function simulated_annealing!(mc::MonteCarlo, schedule, T0::Float64=1.0, path="", rank::Int=0)
    T = T0
    time = 1
    out = length(path) > 0

    # check if configuration file exists. if not, create it 
    if out
        filename="configuration_$rank.h5"
        rank == 0 && !isdir(string(path)) && mkdir( string(path) )
        if !isfile(string(path,filename))
            println("Creating new file $filename for output on rank $rank")
            initialize_hdf5(string(path,filename), mc)
        end
    end

    while T > mc.T
        t = 1
        while t < mc.parameters.t_thermalization
            overrelaxation!(mc.lattice)
            if t % mc.parameters.OR == 0
                metropolis!(mc, T)
            end
            t += 1
        end
        T =  schedule(time)
        time +=1
        if out
            println("Lowering temperature to T=$T on rank $rank and writing checkpoint")
            write_MC_checkpoint!(string(path,filename), mc)  
        else
            println("Lowering temperature to T=$T")
        end
    end
end

# aligns spins to local field 
function deterministic_updates!(mc::MonteCarlo)
    sweeps = 1
    while sweeps < mc.parameters.t_thermalization
        point = rand(1:mc.lattice.size) # pick random index
        field = get_local_field(mc.lattice, point)
        set_spin!(mc.lattice.spins, .-field ./ norm(field) .* mc.lattice.S, point)
        sweeps +=1 
    end
end

function parallel_tempering!(mc::MonteCarlo, path="", saveIC=[])
    # initialize MPI parameters 
    rank = 0
    commSize = 1
    enableMPI = false
    out = length(path) > 0

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

    if out
        filename="configuration_$rank.h5"
        rank == 0 && !isdir(string(path)) && mkdir( string(path) )
        # create new file for output if there isn't already one existing 
        if !isfile(string(path,filename))
            println("Creating new file $filename for output on rank $rank")
            initialize_hdf5(string(path,filename), mc)
        end
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
    IC = (length(saveIC) != 0) ? true : false

    if IC
        any(rank .== saveIC) && println("Initializing IC collection on rank $rank")
        any(rank .== saveIC) && !isdir(string(path, "IC_$rank")) && mkdir( string(path, "IC_$rank") )
    end

    # run parallel tempering
    rank == 0 && @printf("Running sweeps on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))

    while mc.sweep < total_sweeps 

        # overrelaxation sweeps
        overrelaxation!(mc.lattice)

        if mc.sweep % mc.parameters.OR == 0
            # local metroppolis updates 
            accepted_local += metropolis!(mc, mc.T)
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
                    write_MC_checkpoint!(string(path,filename), mc)
                end
                if IC
                    timestep = (mc.sweep - mc.parameters.t_thermalization) ÷ mc.parameters.checkpoint_rate
                    if any(rank .== saveIC)
                        write_initial_configuration!(string(path,"IC_$rank/IC_$timestep.h5"), mc)
                    end
                end
            end 

            # update observables 
            if mc.sweep % mc.parameters.probe_rate == 0
                update_observables!(mc,  E)
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
        write_final_observables!(string(path,filename), mc)
    end

    rank == 0 && @printf("Simulation finished on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
    
    return 
end
