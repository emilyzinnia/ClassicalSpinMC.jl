using .Iterators
using Random

mutable struct Lattice{D,N,R}
    S::Real
    spins::Array{Float64,2}
    unit_cell::UnitCell{D}

    size::Int64      # total number of sites 
    shape::NTuple{D, Int64}      # shape of the bravais lattice 
    site_positions::Array{Float64, 2} # site positions 

    interaction_sites::Vector{NTuple{N, Int64}}
    interaction_matrices::Vector{NTuple{N,InteractionMatrix}}
    ring_exchange_sites::Vector{NTuple{R, NTuple{3, Int64}}}
    ring_exchange_tensors::NTuple{R, Array{Float64, 4}}
    field::Vector{Float64}

    Lattice(D,N,R) = new{D,N,R}()
end

# creates a list of site indices
function site_indices(shape::NTuple{D,Int64}, basis::Int64=1) where D 
    ranges = [ 1:N for N in shape]
    sites = [product(1:basis, ranges...)...]
    return sort!(sites)  #sort by first element 
end

function compute_site_positions(uc::UnitCell{D}, size::NTuple{D,Int64}) where D

    N_sites = prod(size) * length(uc.basis)

    #site positions 
    indices = site_indices(size, length(uc.basis))
    as = uc.lattice_vectors
    basis = uc.basis 
    
    site_positions = Array{Float64,2}(undef, D, N_sites)

    for i in 1:N_sites
        #set site positions
        site_positions[:, i] =  sum( [ (indices[i][d+1]-1) * as[d] for d in 1:D])
        site_positions[:, i] .+= basis[indices[i][1]]
    end

    return site_positions
end

#wrapper for Lattice object 
function lattice(size::NTuple{D,Int64}, uc::UnitCell{D}, initialCondition::Symbol=:random, 
                S::Real=1/2, field::Vector{Float64}=zeros(Float64, 3)) where D
    
    if length(uc.basis) == 0
        addBasisSite!(uc, zeros(Float64, D))
    end

    N_sites = prod(size) * length(uc.basis)
    spins = Array{Float64, 2}(undef, 3, N_sites)

    # initialize spins
    if initialCondition == :random
        for i in 1:N_sites
            set_spin!(spins, random_spin_orientation(S), i)      
        end
    elseif initialCondition == :fm
        spin = random_spin_orientation(S)
        for i in 1:N_sites
            set_spin!(spins, spin, i)      
        end
    end

    indices = site_indices(size, length(uc.basis))
    N = length(uc.interactions)
    R = length(uc.ringexchange)

    lat = Lattice(D,N,R)
    lat.S = S
    lat.shape = size
    lat.size = N_sites
    lat.spins = spins 
    lat.unit_cell = uc
    lat.site_positions = compute_site_positions(uc, size)
    lat.field = field

    lat.interaction_sites = Vector{NTuple{N, Int64}}(undef, 0)
    lat.interaction_matrices = Vector{NTuple{N,InteractionMatrix}}(undef, 0)

    # if no ring exchange terms, make empty matrix 
    lat.ring_exchange_sites = Vector{NTuple{R, NTuple{3, Int64}}}(undef, 0)
    
    # get interactions for each site 
    interactions = uc.interactions
    ring = uc.ringexchange
    lat.ring_exchange_tensors = tuple([ring[r][1] for r=1:R ]...)

    s_ = Vector{Int64}(undef, N)
    M_ = Vector{InteractionMatrix}(undef, N)
    r_ = Vector{NTuple{3, Int64}}(undef, R)
    
    for i in 1:N_sites
        index = indices[i]
        
        #for each interaction term, obtain interaction matrix and index 
        for term in 1:N
            b1, b2, M, offset = interactions[term]
            if (b1 == index[1]) || (b1 == b2)
                bj = b2 
                sign = 1 
            else
                bj = b1 
                sign = -1
                M = transposeJ(M)
            end

            # new_ind = mod.( index[2:end].+ (sign.*offset) .-1, size) .+1
            new_ind = mod.( index[2:end].+ offset .-1, size) .+1
            j = findfirst(x->x == (b2, new_ind...), indices)
            s_[term] = j
            M_[term] = M
        end

        push!(lat.interaction_sites, tuple(s_...))
        push!(lat.interaction_matrices, tuple(M_...))


        # for each ring exchange term, find neighbours and equivalent interaction tensor
        for term in 1:R
            J, j_offset, k_offset, l_offset = ring[term]
            j = findfirst(x->x == (1, (mod.( index[2:end].+ j_offset .-1, size) .+1)...), indices)
            k = findfirst(x->x == (1, (mod.( index[2:end].+ k_offset .-1, size) .+1)...), indices)
            l = findfirst(x->x == (1, (mod.( index[2:end].+ l_offset .-1, size) .+1)...), indices)
            r_[term] = (j, k, l)
        end

        push!(lat.ring_exchange_sites, tuple(r_...))
    end 

    return lat
end

function get_spin(spins::Array{Float64,2}, point::Int64)::NTuple{3, Float64}
    @inbounds return (spins[1, point], spins[2, point], spins[3, point])
end

function set_spin!(spins::Array{Float64,2}, newspin::NTuple{3, Float64}, point::Int64)
    spins[1, point] = newspin[1]
    spins[2, point] = newspin[2]
    spins[3, point] = newspin[3]
end

# pick random point on sphere in spin space (Sx, Sy, Sz)
function random_spin_orientation(S::Real, rng=Random.GLOBAL_RNG)::NTuple{3, Float64}
    phi = 2.0 * pi * rand(rng)
    z = 2.0 * rand(rng) - 1.0;
    r = sqrt(1.0 - z*z)
    return S .* (r*cos(phi), r*sin(phi), z)
end

function get_interaction_sites(lat::Lattice{D,N,R}, point::Int64)::NTuple{N, Int64} where{D,N,R}
    return lat.interaction_sites[point]
end

function get_interaction_matrices(lat::Lattice{D,N,R}, point::Int64)::NTuple{N,InteractionMatrix} where{D,N,R}
    return lat.interaction_matrices[point]
end

function get_ring_exchange_sites(lat::Lattice{D,N,R}, point::Int64)::NTuple{R, NTuple{3, Int64}} where{D,N,R}
    return lat.ring_exchange_sites[point]
end