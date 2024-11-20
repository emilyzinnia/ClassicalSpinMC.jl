using LinearAlgebra

function reciprocal(a1::T, a2::T) where T<:Vector{Float64}
    mag = 2pi  / (a1[1]*a2[2] - a1[2]*a2[1])
    b1 = mag .* [a2[2], -a2[1]]
    b2 = mag .* [-a1[2], a1[1]]
    return (b1, b2) 
end

function reciprocal(a1::T, a2::T, a3::T) where T<:Vector{Float64}
    mag = 2pi  / dot(a1, cross(a2, a3)) 
    b1 = mag .* cross(a2, a3)
    b2 = mag .* cross(a3, a1)
    b3 = mag .* cross(a1, a2)
    return (b1, b2, b3) 
end

""" 
Get allowed wavevectors given the shape of the lattice (tuple). 
min, max sets the boundaries in reciprocal space in multiples of N in each dimension. 
min=0, max=1 corresponds to one copy of the FBZ generated based on the reciprocal lattice vectors. 
"""
function get_allowed_wavevectors(uc::UnitCell{D}, shape::NTuple{D, Int}; min::Int=0, max::Int=1) where D
    reciprocal_vectors = reciprocal(uc.lattice_vectors...) 
    rs = [collect(min*dim:max*dim)/dim for dim in shape] # steps to take along each dimension
    steps = collect.([product(rs...)...])
    allowed_ks = hcat(reciprocal_vectors...) * hcat(steps...) 
    return allowed_ks
end

#FIXME: untested 
function get_k_plane(uc::UnitCell{D}, shape::NTuple{D, Int}; min::Int=-2, max::Int=2) where D
    # plane has to be an array of ones and zeros 
    ks = get_allowed_wavevectors(uc, shape, min=min, max=max)
    temp = copy(ks)
    bounds = broadcast(&, [ 
        collect( round(min*2pi, digits=6) .<= round.(temp[i,:], digits=6) .<= round(max*2pi,digits=6)) 
        for i=1:size(ks)[1] ]...)
    new = temp[:, findall(bounds)]
    return new
end

function get_k_path(uc::UnitCell{D}, direction::Array{Float64,1}, shape::NTuple{D, Int},
                    min::Int=-2, max::Int=2) where D
    ks = get_allowed_wavevectors(uc, shape, min=min, max=max)
    direction ./= norm(direction)

    # get all momenta parallel to direction
    ks_ = ks .- direction
    cond = [ round( abs(dot(direction, ks_[:, k]/norm(ks_[:, k])) ), digits=6) == 1.0 for k=1:size(ks)[2] ]
    temp = ks[:,findall(cond)]

    # get all momenta within the boundaries specified
    bounds = broadcast(&, [ 
        collect( round(min*2pi, digits=6) .<= round.(temp[i,:], digits=6) .<= round(max*2pi,digits=6)) 
        for i=1:size(ks)[1] ]...)
    new = temp[:, findall(bounds)]
    return new
end

"""
Get path in momentum space along high symmetry points 
"""
function get_k_path(UC::UnitCell{D}, hsp::Dict{String, Vector{Float64}}, path::Array{String, 1}, shape::NTuple{D, Int}) where D
    ks = get_allowed_wavevectors(UC, shape)
    kpath = hsp[path[1]] 
    point_count = zeros(Int64, length(path))   # keep track of the indices where the direction changes 
    path_ = copy(path)
    pop!(path_)

    # for each point in k path get wavevectors
    for (ind, point) in enumerate(path_)
        p1 = hsp[point] 
        p2 = hsp[path[ind+1]]

        line = (p2-p1) / norm(p2-p1)
        ks_ = ks .- p1

        # get all momenta parallel to p2-p1
        cond = [ round( abs(dot(line, ks_[:, k]/norm(ks_[:, k])) ), digits=6) == 1.0 for k=1:size(ks)[2] ]
        temp = ks[:, findall(cond)]  # store allowed momenta in temporary vector  
        
        lower = min.(p1, p2)
        upper = max.(p1, p2)
        
        # get all momenta between p1 and p2 
        bounds = broadcast(&, [ 
            collect( round(lower[i], digits=6) .<= round.(temp[i,:], digits=6) .<= round(upper[i], digits=6)) 
            for i=1:size(ks)[1] ]...)

        new_path = temp[:, findall(bounds)]

        # if the order between p1 and p2 were swapped, swap it back 
        if round.(new_path[:,end], digits=5) != round.(p2, digits=5)
            new_path = new_path[:, end:-1:1]
        end

        kpath = hcat(kpath, new_path) # add kpath 
        point_count[ind+1] = size(new_path)[2] + point_count[ind]
    end
    return point_count, kpath
end