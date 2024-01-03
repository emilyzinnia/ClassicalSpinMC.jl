using .Iterators
using Random

mutable struct Lattice{D,N2,N3,N4}
    S::Real
    spins::Array{Float64,2}
    unit_cell::UnitCell
    bc::String

    size::Int64      # total number of sites 
    shape::NTuple{D, Int64}      # shape of the bravais lattice 
    site_positions::Array{Float64, 2} # site positions 

    # Hamiltonian interaction lookups 
    onsite::Vector{InteractionMatrix}
    bilinear_sites::Vector{NTuple{N2, Int64}}
    bilinear_matrices::Vector{NTuple{N2,InteractionMatrix}}
    cubic_sites::Vector{NTuple{N3, NTuple{2, Int64}}}
    cubic_tensors::NTuple{N3, Array{Float64, 3}}
    quartic_sites::Vector{NTuple{N4, NTuple{3, Int64}}}
    quartic_tensors::NTuple{N4, Array{Float64, 4}}
    field::Vector{NTuple{3,Float64}}
    Lattice(D,N2,N3,N4) = new{D,N2,N3,N4}()
end

"""
Creates a 1D array of site indices sorted by basis, row, then column.
"""
function site_indices(shape::NTuple{D,Int64}, basis::Int64=1) where D 
    ranges = [ 1:N for N in shape]
    sites = [product(1:basis, ranges...)...]
    return sort!(sites)  #sort by first element 
end

"""
Compute lattice site positions. 
"""
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

"""
Wrapper for creating Lattice object. 

# Arguments
- `size::NTuple{D,Int64}`: dimensions of lattice for each unit cell, e.g. (4,4) for a 4x4 lattice 
- `uc::UnitCell{D}`: UnitCell object. 
- `S::Real=1/2`: magnitude of spin vector. 

# Keyword Arguments
- `bc::String="periodic"`: boundary conditions; can either be "open" or "periodic". open bc currently only implemented for all boundaries, cannot be partially open/periodic. 
- `initialCondition::Symbol=:random`: all spins start out in a random configuration. can be `:random` or `:fm` where all spins are aligned in an arbitrary direction. 
"""
function Lattice(size::NTuple{D,Int64}, uc::UnitCell{D}, 
                S::Real=1/2; bc::String="periodic", initialCondition::Symbol=:random) where D
    
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
    N2 = length(uc.bilinear)
    N3 = length(uc.cubic)
    N4 = length(uc.quartic)

    lat = Lattice(D,N2,N3,N4)
    lat.S = S
    lat.bc = bc 
    lat.shape = size
    lat.size = N_sites
    lat.spins = spins 
    lat.site_positions = compute_site_positions(uc, size)
    lat.unit_cell = uc 

    function BC(index, offset)
        if bc == "periodic"
            return mod.( index[2:end].+ (offset) .-1, size) .+1
        elseif bc == "open"
            return index[2:end].+ (offset) 
        else
            return error("Invalid boundary condition option")
        end
    end

    ########################################
    # initialize interaction lookup tables 
    ########################################
    # set Zeeman field 
    # if no field specified, set each field to zero on each sublattice 
    # create a matrix containing the field for each basis vector in order 
    lat.field = Vector{NTuple{3,Float64}}(undef, 0)
    lat.onsite = Vector{InteractionMatrix}(undef, 0)

    f_indices = [uc.field[i][1] for i in 1:length(uc.field)]
    f_ = [uc.field[i][2] for i in 1:length(uc.field)]
    field = Matrix{Float64}(undef, 3, length(uc.basis))

    onsite_indices = [uc.onsite[i][1] for i in 1:length(uc.onsite)]
    onsite_matrices = [uc.onsite[i][2] for i in 1:length(uc.onsite)]
    onsite_ = Vector{InteractionMatrix}(undef, length(uc.basis))

    for i in 1:length(uc.basis)
        if !(i in f_indices)
            field[:,i] .= [0.0, 0.0, 0.0]
        else
            field[:,f_indices[i]] .= f_[i]
        end

        if !(i in onsite_indices)
            onsite_[i] = InteractionMatrix(zeros(Float64, 3, 3))
        else
            onsite_[onsite_indices[i]] = onsite_matrices[i]
        end 
    end

    # initialize interaction terms 
    lat.bilinear_sites = Vector{NTuple{N2, Int64}}(undef, 0)
    lat.bilinear_matrices = Vector{NTuple{N2,InteractionMatrix}}(undef, 0)
    lat.cubic_sites = Vector{NTuple{N3, NTuple{2, Int64}}}(undef, 0)
    lat.quartic_sites = Vector{NTuple{N4, NTuple{3, Int64}}}(undef, 0)
    
    # get interactions for each site 
    bilinear = uc.bilinear
    cubic = uc.cubic
    quartic = uc.quartic

    printstyled("WARNING: "; color = :yellow)
    println("Cubic and quartic interactions untested for unit cells with more than one basis site. Use with caution.")
    lat.cubic_tensors = tuple([cubic[r][1] for r=1:N3 ]...)
    lat.quartic_tensors = tuple([quartic[r][1] for r=1:N4 ]...)

    s2_ = Vector{Int64}(undef, N2)                  # bilinear site indices
    M2_ = Vector{InteractionMatrix}(undef, N2)      # bilinear interaction matrices
    s3_ = Vector{NTuple{2, Int64}}(undef, N3)       # cubic site indices
    s4_ = Vector{NTuple{3, Int64}}(undef, N4)       # quartic site indices

    for i in 1:N_sites
        index = indices[i]
        
        # add local zeeman coupling 
        push!(lat.field, tuple(field[:,index[1]]...))
        push!(lat.onsite, onsite_[index[1]])

        # for each interaction term, obtain interaction matrix and index 
        for term in 1:N2
            b1, b2, M, offset = bilinear[term]
            if b1 == b2
                bj = index[1] 
                sign = 1
            elseif (b1 == index[1]) 
                bj = b2 
                sign = 1 
            else
                bj = b1 
                sign = -1
                M = transposeJ(M)
            end
            new_ind = BC(index, sign.*offset)
            j = findfirst(x->x == (bj, new_ind...), indices)
            if !isnothing(j)
                s2_[term] = j
                M2_[term] = M
            else # open BC 
                s2_[term] = 1
                M2_[term] = InteractionMatrix(zeros(Float64, 3, 3))
            end
        end
        push!(lat.bilinear_sites, tuple(s2_...))
        push!(lat.bilinear_matrices, tuple(M2_...))
        
        # for each cubic term, find neighbours and equivalent interaction tensor
        for term in 1:N3
            b1, b2, b3, J, j_offset, k_offset = cubic[term]
            j = findfirst(x->x == (b2, BC(index, j_offset)...), indices)
            k = findfirst(x->x == (b3, BC(index, k_offset)...), indices)
            if isnothing(j) | isnothing(k) 
                error("Open BC not implemented for cubic")
            end
            s3_[term] = (j, k)
        end
        push!(lat.cubic_sites, tuple(s3_...))

        # for each quartic term, find neighbours and equivalent interaction tensor
        for term in 1:N4
            b1, b2, b3, b4, J, j_offset, k_offset, l_offset = quartic[term]
            j = findfirst(x->x == (b2, BC(index, j_offset)...), indices)
            k = findfirst(x->x == (b3, BC(index, k_offset)...), indices)
            l = findfirst(x->x == (b4, BC(index, l_offset)...), indices)
            if isnothing(j) | isnothing(k) | isnothing(l)
                error("Open BC not implemented for quartic")
            end
            s4_[term] = (j, k, l)
        end
        push!(lat.quartic_sites, tuple(s4_...))
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

"""
Pick random point on sphere in spin space (Sx, Sy, Sz)
"""
function random_spin_orientation(S::Real, rng=Random.GLOBAL_RNG)::NTuple{3, Float64}
    phi = 2.0 * pi * rand(rng)
    z = 2.0 * rand(rng) - 1.0;
    r = sqrt(1.0 - z*z)
    return S .* (r*cos(phi), r*sin(phi), z)
end

function get_onsite(lat::Lattice{D,N2,N3,N4}, point::Int64)::InteractionMatrix where{D,N2,N3,N4}
    return lat.onsite[point]
end

function get_field(lat::Lattice{D,N2,N3,N4}, point::Int64)::NTuple{3, Float64} where{D,N2,N3,N4}
    return lat.field[point]
end

function get_bilinear_sites(lat::Lattice{D,N2,N3,N4}, point::Int64)::NTuple{N2, Int64} where{D,N2,N3,N4}
    return lat.bilinear_sites[point]
end

function get_bilinear_matrices(lat::Lattice{D,N2,N3,N4}, point::Int64)::NTuple{N2,InteractionMatrix} where{D,N2,N3,N4}
    return lat.bilinear_matrices[point]
end

function get_cubic_sites(lat::Lattice{D,N2,N3,N4}, point::Int64)::NTuple{N3, NTuple{2, Int64}} where{D,N2,N3,N4}
    return lat.cubic_sites[point]
end

function get_quartic_sites(lat::Lattice{D,N2,N3,N4}, point::Int64)::NTuple{N4, NTuple{3, Int64}} where{D,N2,N3,N4}
    return lat.quartic_sites[point]
end
