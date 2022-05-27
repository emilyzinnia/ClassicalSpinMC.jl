using ProfileView
using Profile
using Plots
using Pkg
Pkg.activate("ClassicalSpinMC")
using ClassicalSpinMC

include("honeycomb.jl")

function plot_lattice(lat)
    plotly()
    pos = lat.site_positions 
    a1, a2 = lat.unit_cell.lattice_vectors
    
    scatter(pos[1,:], pos[2,:], aspect_ratio=:equal, 
    series_annotations = text.(1:length(pos[1,:]), :bottom))

    right = pos .+ a1 * lat.shape[1]
    left = pos .- a1 * lat.shape[1]
    up = pos .+ a2 * lat.shape[2]
    down = pos .- a2 * lat.shape[2]
    scatter!( right[1,:], right[2,:], aspect_ratio=:equal, 
    series_annotations = text.(1:length(pos[1,:]), :bottom))
    scatter!( left[1,:], left[2,:], aspect_ratio=:equal, 
    series_annotations = text.(1:length(pos[1,:]), :bottom))
    scatter!( up[1,:], up[2,:], aspect_ratio=:equal, 
    series_annotations = text.(1:length(pos[1,:]), :bottom))
    scatter!( down[1,:], down[2,:], aspect_ratio=:equal, 
    series_annotations = text.(1:length(pos[1,:]), :bottom))
end

function energy_benchmark()
    L = 4
    S = 1
    K = -1.0
    G = 0.2
    Gp = -0.02
    field = 0.1 * [1,1,-2]/sqrt(6)

    H = Honeycomb()
    lattice_params = Dict("K"=>K, "G"=>G, "Gp"=>Gp)
    addInteractionsKitaev!(H, lattice_params)
    lat = lattice( (L,L), H, :random, S, field) 

    T0 = 1.0
    T=1e-7
    t_sweep= Int(1e4)
    t_measurement= Int(5e6)
    probe_rate=10
    swap_rate=10
    overrelaxation=10
    report_interval = Int(1e4)
    checkpoint_rate=500
    params = SimulationParameters(t_sweep, t_measurement, probe_rate, 
                              swap_rate, overrelaxation, report_interval, checkpoint_rate)

    mc = MonteCarlo(T, lat, params) 
    println("Running simulated annealing")
    simulated_annealing!(mc, x -> T0*0.9^x , T0)
    println("Deterministic updates")
    deterministic_updates!(mc)

    println(energy_density(mc.lattice))
end



