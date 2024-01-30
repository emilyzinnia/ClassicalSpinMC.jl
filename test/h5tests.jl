using HDF5
using ClassicalSpinMC: dump_unit_cell!, read_unit_cell, dump_lattice!, read_lattice
using Base.Filesystem

@testset "Dumping Metadata" begin 
    U = Square()
    lat = Lattice( (2,2), U, S=1.0)
    h5open(".test.h5", "w") do f 
        dump_unit_cell!(f,U)
        dump_lattice!(f,lat)
    end 

    @testset "Unit Cell" begin
        R = read_unit_cell(".test.h5")
        @test U.n == R.n 
        @test U.lattice_vectors == R.lattice_vectors 
        @test U.basis == R.basis
        @test U.field == R.field 
        @test U.onsite == R.onsite
        @test U.bilinear == R.bilinear 
        @test U.cubic == R.cubic 
        @test U.quartic == R.quartic
    end

    @testset "Lattice" begin
        L = read_lattice(".test.h5")
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
    
    Filesystem.unlink(".test.h5")
end
