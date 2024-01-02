
struct InteractionMatrix
    m11::Float64
    m12::Float64
    m13::Float64
    m21::Float64
    m22::Float64
    m23::Float64
    m31::Float64
    m32::Float64
    m33::Float64
end

function InteractionMatrix(J::T) where T<:AbstractArray
    return InteractionMatrix(J[1, 1], J[1, 2], J[1, 3],
                             J[2, 1], J[2, 2], J[2, 3],
                             J[3, 1], J[3, 2], J[3, 3])
end

# return Matrix object from InteractionMatrix 
function Matrix(J::InteractionMatrix)
    return [J.m11 J.m12 J.m13 
            J.m21 J.m22 J.m23
            J.m31 J.m32 J.m33]
end 

function transposeJ(J::InteractionMatrix)
    return InteractionMatrix(J.m11, J.m21, J.m31, 
                             J.m12, J.m22, J.m32,
                             J.m13, J.m23, J.m33)
end
