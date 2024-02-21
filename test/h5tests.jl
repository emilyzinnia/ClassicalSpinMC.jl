using HDF5
using ClassicalSpinMC: dump_unit_cell!, read_unit_cell, dump_metadata!, read_lattice
using Base.Filesystem

@testset "Dumping Metadata" begin 
    U = Square()
    lat = Lattice( (2,2), U, 1.0)
    params = Dict("t_thermalization"=> Int(1e4),"overrelaxation_rate"=>10, "t_deterministic"=>Int(1e7))
    mc = MonteCarlo(1.0, lat, params)
    h5open(".test.h5", "w") do f 
        dump_metadata!(f, mc)
    end 

    f = h5open(".test.h5", "r")

    @testset "Unit Cell" begin
        R = read_unit_cell(f)
        @test U.lattice_vectors == R.lattice_vectors 
        @test U.basis == R.basis
        @test U.field == R.field 
        @test U.onsite == R.onsite
        @test U.bilinear == R.bilinear 
        @test U.cubic == R.cubic 
        @test U.quartic == R.quartic
    end

    @testset "Lattice" begin
        L = read_lattice(f)
        @test lat.S == L.S 
        @test lat.bc == L.bc
        @test lat.size == L.size 
        @test lat.shape == L.shape 
        @test lat.site_positions == L.site_positions 
        @test lat.onsite == L.onsite 
        @test lat.bilinear_sites == L.bilinear_sites 
        @test lat.bilinear_matrices == L.bilinear_matrices 
        @test lat.cubic_sites == L.cubic_sites 
        @test lat.cubic_tensors == L.cubic_tensors 
        @test lat.quartic_sites == L.quartic_sites 
        @test lat.quartic_tensors == L.quartic_tensors 
        @test lat.field == L.field  
    end
    
    close(f)
    Filesystem.unlink(".test.h5")
end
