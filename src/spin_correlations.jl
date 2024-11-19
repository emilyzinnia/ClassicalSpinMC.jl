# spin_correlations.jl

"""
Computes equal time spin correlations in momentum space 
"""
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

"""
Runner for processing a large batch of SSF calculations for a given parameter set 
"""
function runEqualTimeStructureFactor!(path, lat::Lattice, ks::Matrix{Float64}, override=false)
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
    N_IC = length(readdir(path)) # number of configurations 
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


"""
Averages over all configurations computed from runEqualTimeStructureFactor!
"""
function compute_equal_time_structure_factor(path::String, dest::String)
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
