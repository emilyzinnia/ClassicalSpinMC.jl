using LinearAlgebra

@testset "Norm Spin test" begin
    U = Square()
    lat = Lattice((2,2), U, 1.0)
    @test all(round.(norm.(eachcol(lat.spins)), digits=9) .== 1.0)
end

@testset "Energy test" begin
    @testset "Zeeman" begin
        U = Square()
        h = [1.0, 0.0, 0.0]
        addZeemanCoupling!(U, 1, h)
        lat = Lattice((1,1), U, 1.0)
        lat.spins .= [1.0, 0.0, 0.0]

        @test -1.0 == total_energy(lat)
        @test tuple((-h)...) == get_local_field(lat, 1)
    end

    @testset "Bilinear" begin
        U = Square()
        J = -1.0 * collect(1.0I(3))
        addBilinear!(U, 1, 1, J, (1, 0))
        addBilinear!(U, 1, 1, J, (-1, 0))
        addBilinear!(U, 1, 1, J, (0, 1))
        addBilinear!(U, 1, 1, J, (0, -1))
        lat = Lattice((2,2), U, 1.0)
        lat.spins .= [1.0, 0.0, 0.0]
        @test -2.0 == total_energy(lat) / lat.size 
    end
end