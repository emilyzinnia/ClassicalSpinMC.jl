using Pkg
Pkg.activate("ClassicalSpinMC")
using ClassicalSpinMC
using MPI

include("honeycomb.jl")
include("input_file.jl")

# initialize MPI
MPI.Initialized() || MPI.Init()
commSize = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# get temperatures 
temp = exp10.(range(log10(Tmin), stop=log10(Tmax), length=commSize) ) 
T = temp[rank+1]

# read command line arguments 
path = ARGS[1]

# generate lattice and mc runner 
H = Honeycomb()
lattice_params = Dict("J1z"=>J1z, "J1xy"=>J1xy, "J3xy"=>J3xy, "J3z"=>J3z, "D"=>D, "E"=>E)
addInteractionsCartesian!(H, lattice_params)
lat = lattice( (L,L), H; S=S, field=field) 

params = SimulationParameters(t_sweep, t_measurement, probe_rate, 
                              swap_rate, overrelaxation, report_interval, checkpoint_rate)
mc = MonteCarlo(T, lat, params)

#initialize hdf5 for output
filename = "configuration_$rank.h5"
rank == 0 && !isdir(string(path)) && mkdir( string(path) )
if !isfile(string(path,filename))
    println("Creating new file $filename for output on rank $rank")
    initialize_hdf5(string(path,filename), mc, lattice_params)
end

# perform MC tasks 
simulated_annealing!(mc, x -> T0*0.9^x , T0)
deterministic_updates!(mc)
parallel_tempering!(mc, path, [])
