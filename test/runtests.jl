using ClassicalSpinMC
using Test 

@testset "ClassicalSpinMC tests" begin
    @testset "Lattice tests" begin
        include("latticetests.jl")
    end

    @testset "Monte Carlo tests" begin
        include("mctests.jl")
    end

    @testset "HDF5 tests" begin
        include("h5tests.jl")
    end
end