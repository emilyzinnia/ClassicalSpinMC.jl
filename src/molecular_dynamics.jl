using DifferentialEquations
using Dates
using BinningAnalysis
using HDF5
using ProgressMeter

struct MDbuffer
    tstep::Float64
    tmin::Float64
    tmax::Float64
    tt::Array{Float64, 1}
    freq::Array{Float64, 1}
    ks::Array{Float64, 2}
    alpha::Float64
end

function get_tt_freq(tstep::Float64=0.01, tmin::Float64=0.0, tmax::Float64=0.5)
    tt = collect(tmin:tstep:tmax)
    N_time = length(tt)
    N = iseven(N_time) ? N_time : N_time+1
    freq = [ n/(N*tstep) for n in 0:N ] 
    return tt, freq
end 

function MD_buffer(ks::Array{Float64}, tstep::Float64=0.01, tmin::Float64=0.0, tmax::Float64=100.0, alpha::Float64=0.0)::MDbuffer
    # frequencies
    tt, freq = get_tt_freq(tstep, tmin, tmax)
    return MDbuffer(tstep, tmin, tmax, tt, freq, ks, alpha)
end

function timeEvolve!(du, u, p, t)
    N = p[1] # size of the lattice 
    interaction_sites = p[2] # interaction sites 
    interaction_matrices = p[3] # interaction matrices
    hs = p[4] # field 
    alpha = p[5] # damping parameter 
    for i=1:N
        js = interaction_sites[i]
        Js = interaction_matrices[i]
        h = hs[i]
        Hx = 0.0
        Hy = 0.0
        Hz = 0.0
        # calculate local effective field 
        for n in eachindex(js)
            J = Js[n]
            j = js[n]
            ux = u[3j-2]
            uy = u[3j-1]
            uz = u[3j]

            Hx += 2*(J.m11 * ux + J.m12 * uy + J.m13 * uz)
            Hy += 2*(J.m21 * ux + J.m22 * uy + J.m23 * uz)
            Hz += 2*(J.m31 * ux + J.m32 * uy + J.m33 * uz)
        end
        # field term 
        Hx += -h[1]
        Hy += -h[2]
        Hz += -h[3] 
        # components are stored in multiples of i
        px= (u[3i-1] * Hz - u[3i] * Hy)
        py= (u[3i] * Hx - u[3i-2] * Hz)
        pz= (u[3i-2] * Hy - u[3i-1] * Hx)
        du[3i-2] = (u[3i-1] * (Hz + alpha*pz) - u[3i] * (Hy + alpha*py) )
        du[3i-1] =(u[3i] * (Hx + alpha*px) - u[3i-2] * (Hz + alpha*pz))
        du[3i] = (u[3i-2] * (Hy + alpha*py) - u[3i-1] * (Hx + alpha*px))
    end
end

function compute_time_evolution(lat::Lattice, md::MDbuffer, alg=Tsit5(), tol::Float64=1e-7)
    # time evolve the spins 
    s0 = vcat(lat.spins...)   # flatten to vector of (Sx1, Sy1, Sz1...)
    params = [lat.size, lat.interaction_sites, lat.interaction_matrices, lat.field, md.alpha]
    ks = md.ks
    N_k = size(ks)[2]
    pos = lat.site_positions
    omega = md.freq 
    Sqw = zeros(ComplexF64, 3N_k, length(omega))
    spins = zeros(Float64, 3, lat.size)

    function perform_measurements!(integrator)
        t = integrator.t
        spins[1,:] .= integrator.u[3 * collect(1:lat.size) .- 2] #sx
        spins[2,:] .= integrator.u[3 * collect(1:lat.size) .- 1] #sy
        spins[3,:] .= integrator.u[3 * collect(1:lat.size) ]     #sz

        for n=1:N_k
            phase = exp.(-im * transpose(ks[:, n]) * pos)
            sqx = (phase * spins[1,:])[1] 
            sqy = (phase * spins[2,:])[1] 
            sqz = (phase * spins[3,:])[1] 
            for w in 1:length(omega)
                Sqw[3n-2, w] += sqx * exp(im * omega[w] * t)
                Sqw[3n-1, w] += sqy * exp(im * omega[w] * t)
                Sqw[3n  , w] += sqz * exp(im * omega[w] * t)
            end
        end
    end

    # perform_measurements!(integrator) = push!(measurements, integrator.u) 
    prob = ODEProblem(timeEvolve!, s0, (md.tmin, md.tmax), params)
    cb = PresetTimeCallback(md.tt, perform_measurements!)
    
    # solve ODE 
    sol = solve(prob, alg, reltol=tol, abstol=tol, callback=cb, dense=false, save_on=false)
    return Sqw[3 * collect(1:N_k) .- 2, :], Sqw[3 * collect(1:N_k) .- 1, :], Sqw[3 * collect(1:N_k) , :] 
end

# do this step in parallel 
function compute_FT_correlations(S_q::NTuple{3, Array{ComplexF64, 2}}, lat::Lattice, md::MDbuffer)
    N_k = size(md.ks)[2]
    Suv = zeros(Float64, 9, N_k, length(md.freq)) # store SSF results
    sx, sy, sz = S_q 
    
    sx_ = conj.(sx)
    sy_ = conj.(sy)
    sz_ = conj.(sz)

    # compute correlations
    Suv[1, :, :] .= real.(sx .* sx_)
    Suv[2, :, :] .= real.(sx .* sy_)
    Suv[3, :, :] .= real.(sx .* sz_)
    Suv[4, :, :] .= real.(sy .* sx_)
    Suv[5, :, :] .= real.(sy .* sy_)
    Suv[6, :, :] .= real.(sy .* sz_)
    Suv[7, :, :] .= real.(sz .* sx_)
    Suv[8, :, :] .= real.(sz .* sy_)
    Suv[9, :, :] .= real.(sz .* sz_)
    return  Suv ./ (2*lat.size*pi)
end

function compute_equal_time_correlations(lat::Lattice, ks::Array{Float64,2})
    N_k = size(ks)[2]
    pos = lat.site_positions
    Suv = Array{Float64, 2}(undef, 9, N_k)
    spins = lat.spins

    x = spins[1,:]
    y = spins[2,:]
    z = spins[3,:]
    
    sx = zeros(ComplexF64, N_k)
    sy = zeros(ComplexF64, N_k)
    sz = zeros(ComplexF64, N_k)

    for n=1:N_k
        phase = exp.(-im * transpose(ks[:, n]) * pos)
        sx[n] = (phase * x)[1] 
        sy[n] = (phase * y)[1] 
        sz[n] = (phase * z)[1] 
    end

    sx_ = conj.(sx)
    sy_ = conj.(sy)
    sz_ = conj.(sz)

    # compute correlations
    Suv[1, :] .= real.(sx .* sx_)
    Suv[2, :] .= real.(sx .* sy_)
    Suv[3, :] .= real.(sx .* sz_)
    Suv[4, :] .= real.(sy .* sx_)
    Suv[5, :] .= real.(sy .* sy_)
    Suv[6, :] .= real.(sy .* sz_)
    Suv[7, :] .= real.(sz .* sx_)
    Suv[8, :] .= real.(sz .* sy_)
    Suv[9, :] .= real.(sz .* sz_)

    return Suv ./ lat.size
end

function runStaticStructureFactor!(path, lat::Lattice, ks::Matrix{Float64}, override=false)
    # initialize MPI parameters 
    rank = 0
    commSize = 1
    enableMPI = false
    
    if MPI.Initialized()
        comm = MPI.COMM_WORLD
        commSize = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        if commSize > 1
            enableMPI = true
        end
    end

    # split computations evenly along all nodes 
    N_IC = length(readdir(path)) # number of initial configurations 
    N_rank = Array{Int64, 1}(undef, commSize)
    N_per_rank = N_IC ÷ commSize 
    N_remaining = N_IC % commSize

    # distribute remaining configurations 
    if (N_remaining != 0) && rank < N_remaining
        N_per_rank += 1
    end 

    N_rank[rank+1] = N_per_rank
    commSize > 1 && MPI.Allgather!(MPI.IN_PLACE, N_rank, 1, MPI.COMM_WORLD) 

    IC = rank == 0 ? 0 : sum(N_rank[1:rank])

    # initialize lattice object 
    for i in 1:N_per_rank
        
        # initialize lattice object from hdf5 file 
        file = string(path, "IC_$IC.h5") 
        read_spin_configuration!(file, lat)

        f = h5open(file, "r+") 
        exists = haskey(f, "spin_correlations")
        close(f)
        if exists && !override
            println("Skipping IC_$IC")
            IC += 1
            continue
        else
            # compute correlations and output
            println("Computing SSF $i/$N_per_rank on rank $rank")
            @time S = compute_equal_time_correlations(lat, ks)

            println("Writing IC $IC to file on rank $rank")
            res = Dict("SSF"=>S, "SSF_momentum"=>ks)
            f = h5open(file, "r+")
            g = haskey(f, "spin_correlations") ? f["spin_correlations"] : create_group(f, "spin_correlations")
            overwrite_keys!(g, res)
            close(f)
        end
        # increment configuration 
        IC += 1
    end

    println("Calculation completed on rank $rank on ", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
end

function runMolecularDynamics!(path, tstep, tmin, tmax, lat::Lattice, ks::Matrix{Float64}, alg=Tsit5(), tol::Float64=1e-7,
                               override=false; alpha::Float64=0.0)
    # initialize MPI parameters 
    rank = 0
    commSize = 1
    enableMPI = false
    
    if MPI.Initialized()
        comm = MPI.COMM_WORLD
        commSize = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        if commSize > 1
            enableMPI = true
        end
    end

    # split computations evenly along all nodes 
    N_IC = length(readdir(path)) # number of initial configurations 
    N_rank = Array{Int64, 1}(undef, commSize)
    N_per_rank = N_IC ÷ commSize 
    N_remaining = N_IC % commSize

    # distribute remaining configurations 
    if (N_remaining != 0) && rank < N_remaining
        N_per_rank += 1
    end 

    N_rank[rank+1] = N_per_rank
    commSize > 1 && MPI.Allgather!(MPI.IN_PLACE, N_rank, 1, MPI.COMM_WORLD) 

    IC = rank == 0 ? 0 : sum(N_rank[1:rank])
    MD = MD_buffer(ks, tstep, tmin, tmax, alpha)

    for i in 1:N_per_rank
        # initialize lattice object from hdf5 file 
        file = string(path, "IC_$IC.h5") 
        read_spin_configuration!(file, lat)
        
        f = h5open(file, "r+") 
        exists = haskey(f, "spin_correlations")
        close(f)
        if exists && !override
            println("Skipping IC_$IC")
            IC += 1
            continue
        else
            println("Time evolving for IC $i/$N_per_rank on rank $rank")
            @time S_t = compute_time_evolution(lat, MD, alg, tol)
            corr = compute_FT_correlations(S_t, lat, MD)

            println("Writing IC $IC to file on rank $rank")
            res = Dict("S_qw"=>corr, "freq"=>MD.freq, "momentum"=>ks)
            h5open(file, "r+") do f
                g = haskey(f, "spin_correlations") ? f["spin_correlations"] : create_group(f, "spin_correlations")
                overwrite_keys!(g, res)
            end
        end
        # increment configuration 
        IC += 1
    end

    println("Calculation completed on rank $rank on ", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
end




function compute_static_structure_factor(path::String, dest::String)
    # initialize LogBinner
    println("Initializing LogBinner in $path")
    files = readdir(path)
    f0 = h5open(string(path, files[1]), "r")
    shape = size(read(f0["spin_correlations/SSF"]) )
    ks = read(f0["spin_correlations/SSF_momentum"])

    SSF = LogBinner(zeros(Float64, shape...))
    close(f0)

    # collecting correlations
    println("Collecting correlations from ", length(files), " files")
    @showprogress for file in files 
        h5open(string(path, file), "r") do f 
            Suv = read(f["spin_correlations/SSF"])
            push!(SSF, Suv)
        end
    end

    # write to configuration file 
    println("Writing to $dest")
    d = h5open(dest, "r+")
    res = Dict(
               "SSF"=>mean(SSF), 
               "SSF_momentum"=>ks)
    g = haskey(d, "spin_correlations") ? d["spin_correlations"] : create_group(d, "spin_correlations")
    overwrite_keys!(g, res)
    close(d)
    println("Done")
end

function compute_dynamic_structure_factor(path::String, dest::String, params::Dict{String, Float64})
    # initialize LogBinner
    println("Initializing LogBinner in $path")
    files = readdir(path)
    f0 = h5open(string(path, files[1]), "r")
    shape = size(read(f0["spin_correlations/S_qw"]) )
    ks = read(f0["spin_correlations/momentum"])
    freq = read(f0["spin_correlations/freq"])
    close(f0)
    DSF = LogBinner(zeros(Float64, shape...))

    # collecting correlations
    println("Collecting correlations from ", length(files), " files")
    @showprogress for file in files 
        h5open(string(path, file), "r") do f 
            Suv = read(f["spin_correlations/S_qw"])
            push!(DSF, Suv)
        end
    end

    # write to configuration file 
    println("Writing to $dest")
    d = h5open(dest, "r+")
    res = Dict("freq"=>freq, "momentum"=>ks,  "S_qw"=>mean(DSF))
    g = haskey(d, "spin_correlations") ? d["spin_correlations"] : create_group(d, "spin_correlations")
    overwrite_keys!(g, params)
    overwrite_keys!(g, res)
    close(d)

    println("Successfully averaged ",length(files), " files")
end



