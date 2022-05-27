#observables.jl
using BinningAnalysis
using LinearAlgebra

mutable struct Observables
    energy::ErrorPropagator{Float64,32}
    magnetization::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    energyTimeSeries::FullBinner{Float64,Vector{Float64}}
    roundtripMarker::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    Observables() = new(ErrorPropagator(Float64), LogBinner(Float64), FullBinner(Float64), LogBinner(Float64) )
end

function get_magnetization(lattice::Lattice)::Float64
    m = (0.0, 0.0, 0.0)
    for i in 1:lattice.size
        m = m .+ get_spin(lattice.spins, i)
    end
    return norm(m)
end

function update_observables!(mc, energy::Float64)
    #measure energy and energy^2
    push!(mc.observables.energy, energy, energy^2 )
    #measure magnetization
    push!(mc.observables.magnetization, get_magnetization(mc.lattice))
end

function std_error_tweak(ep::ErrorPropagator, gradient, lvl = BinningAnalysis._reliable_level(ep))
    sqrt(abs(var(ep, gradient, lvl) / ep.count[lvl]))
end

function specific_heat(mc) 
    ep = mc.observables.energy
    temp = mc.T
    lat = mc.lattice 

    #compute specific heat 
    c(e) = 1/temp^2 * (e[2]-e[1]*e[1]) / lat.size
    ∇c(e) = [-2.0 * 1/temp^2 * e[1] / lat.size, 1/temp^2 / lat.size] 

    heat = mean(ep, c)
    dheat = std_error_tweak(ep, ∇c)

    return heat, dheat
end 