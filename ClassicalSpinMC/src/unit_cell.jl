

struct UnitCell{D}
    lattice_vectors::NTuple{D, Vector{Float64}}
    basis::Vector{Vector{Float64}}

    interactions::Vector{ Tuple{Int64, Int64, InteractionMatrix, NTuple{D, Int64}}}
    ringexchange::Vector{ Tuple{Array{Float64, 4}, NTuple{D, Int64}, NTuple{D, Int64}, NTuple{D, Int64}}}

    UnitCell(as...) = new{length(as)}(as, Vector{Vector{Float64}}(undef,0), 
                    Vector{ Tuple{Int64, Int64, InteractionMatrix, NTuple{length(as), Int64}}}(undef,0),
                    Vector{ Tuple{Array{Float64, 4}, NTuple{length(as), Int64}, NTuple{length(as), Int64}, NTuple{length(as), Int64}}}(undef,0))
end

function addBasisSite!(uc::UnitCell{D}, site::Vector{Float64}) where D
    push!(uc.basis, site)
end

"""Add interaction between basis site b1 and b2, with offset denoting the unit cell offset."""
function addInteraction!(uc::UnitCell{D}, b1::Int64, b2::Int64, M::Matrix{Float64}, 
                         offset::NTuple{D, Int64}=Tuple(zeros(Int64, D))) where D

    if !iszero(M)
        push!(uc.interactions, (b1, b2, InteractionMatrix(M), offset))
    end
end

"""Add general ring exchange between sites with o2, o3, o4 denoting the unit cell offset."""
function addRingExchange!(uc::UnitCell{D}, M::Array{Float64, 4}, 
                          o2::NTuple{D, Int64}=Tuple(zeros(Int64, D)), 
                          o3::NTuple{D, Int64}=Tuple(zeros(Int64, D)),
                          o4::NTuple{D, Int64}=Tuple(zeros(Int64, D))) where D
    length(uc.basis) > 1 && error("Ring exchange not implemented for lattices with more than one basis site")
    push!(uc.ringexchange, (M, o2, o3, o4))
end
