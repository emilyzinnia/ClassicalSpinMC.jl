@testset "Annealing test" begin
    L = 4
    S = 1
    J = 0.0
    K = -1.0
    G = 0.2
    Gp = -0.02
    field = 0.1 * [1,1,1]/sqrt(3)

    # define unit cell object 
    H = Honeycomb()

    # xbond
    Jx = [K+J Gp Gp
        Gp J G
        Gp G J]
    # ybond
    Jy = [J Gp G
        Gp K+J Gp 
        G Gp J]
    # zbond
    Jz = [J G Gp 
        G J Gp
        Gp Gp K+J]

    # nearest neighbour interactions 
    addBilinear!(H, 1, 2, Jx, (0, -1)) #x-bond 
    addBilinear!(H, 1, 2, Jy, (1, -1)) #y-bond 
    addBilinear!(H, 1, 2, Jz, (0, 0 )) #z-bond 

    # add Zeeman coupling 
    addZeemanCoupling!(H, 1, field)
    addZeemanCoupling!(H, 2, field)

    # create lattice object 
    lat = Lattice( (L,L), H, S) 
    
    # define MC params 
    T0 = 1.0 # starting temperature for simulated annealing
    T = 1e-7 # target temperature 
    params = Dict("t_thermalization"=> Int(1e4),"overrelaxation_rate"=>10, "t_deterministic"=>Int(1e6))
    
    @testset "Metropolis" begin
        # create MC object 
        mc = MonteCarlo(T, lat, params) 
        simulated_annealing!(mc, x -> T0*0.9^x , T0)
        deterministic_updates!(mc)
        E = energy_density(mc.lattice)
        @test -0.6444 == round(E, digits=4)
    end

    @testset "MetropolisAdaptive" begin
        # create MC object 
        mc = MonteCarlo(T, lat, params) 
        simulated_annealing!(mc, x -> T0*0.9^x , T0, alg=MetropolisAdaptive())
        E = energy_density(mc.lattice)
        @test -0.6444 == round(E, digits=4)
    end
end
