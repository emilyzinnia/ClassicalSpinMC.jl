

struct UnitCell{D}
    lattice_vectors::NTuple{D, Vector{Float64}}
    basis::Vector{Vector{Float64}}
    field::Vector{ Tuple{Int64, Vector{Float64}} }
    bilinear::Vector{ Tuple{Int64, Int64, InteractionMatrix, NTuple{D, Int64}}}
    cubic::Vector{  Tuple{Int64, Int64, Int64, Array{Float64, 3}, NTuple{D, Int64}, NTuple{D, Int64}}}
    quartic::Vector{ Tuple{Int64, Int64, Int64, Int64, Array{Float64, 4}, NTuple{D, Int64}, NTuple{D, Int64}, NTuple{D, Int64}}}

    UnitCell(as...) = new{length(as)}(as, Vector{Vector{Float64}}(undef,0), 
                    Vector{ Tuple{Int64, Vector{Float64}}}(undef, 0),
                    Vector{ Tuple{Int64, Int64, InteractionMatrix, NTuple{length(as), Int64}}}(undef,0),
                    Vector{ Tuple{Int64, Int64, Int64, Array{Float64, 3}, NTuple{length(as), Int64}, NTuple{length(as), Int64}}}(undef,0),
                    Vector{ Tuple{Int64, Int64, Int64, Int64, Array{Float64, 4}, NTuple{length(as), Int64}, NTuple{length(as), Int64}, NTuple{length(as), Int64}}}(undef,0))
end

function addBasisSite!(uc::UnitCell{D}, site::Vector{Float64}) where D
    push!(uc.basis, site)
end

"""Add Zeeman coupling on basis site b1"""
function addZeemanCoupling!(uc::UnitCell{D}, b1::Int64, h::Vector{Float64}) where D
    push!(uc.field, (b1, h))
end

"""Add bilinear interaction between basis site b1 and b2, with offset denoting the unit cell offset."""
function addBilinear!(uc::UnitCell{D}, b1::Int64, b2::Int64, M::Matrix{Float64}, 
                         offset::NTuple{D, Int64}=Tuple(zeros(Int64, D))) where D

    if !iszero(M)
        push!(uc.bilinear, (b1, b2, InteractionMatrix(M), offset))
    end
end

"""Add general cubic interaction between basis sites b1, b2, and b3 with o2, o3 denoting the unit cell offset."""
function addCubic!(uc::UnitCell{D}, b1::Int64, b2::Int64, b3::Int64, M::Array{Float64, 3}, 
                          o2::NTuple{D, Int64}=Tuple(zeros(Int64, D)), 
                          o3::NTuple{D, Int64}=Tuple(zeros(Int64, D))) where D
    push!(uc.cubic, (b1, b2, b3, M, o2, o3))
end

"""Add general quartic interaction between sites with o2, o3, o4 denoting the unit cell offset."""
function addQuartic!(uc::UnitCell{D}, b1::Int64, b2::Int64, b3::Int64, b4::Int64, M::Array{Float64, 4}, 
                          o2::NTuple{D, Int64}=Tuple(zeros(Int64, D)), 
                          o3::NTuple{D, Int64}=Tuple(zeros(Int64, D)),
                          o4::NTuple{D, Int64}=Tuple(zeros(Int64, D))) where D
    push!(uc.quartic, (b1, b2, b3, b4, M, o2, o3, o4))
end
