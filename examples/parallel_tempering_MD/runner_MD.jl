using ClassicalSpinMC
using MPI 
using HDF5

include("input_file.jl")
include("pyrochlore.jl")

# simulation parameters 
dt = 0.05
tmax = 60.0 

# initialize MPI 
MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD
commSize = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

# read command line arguments 
path = string(ARGS[1]) # working path with trailing backslash 
ic_path = string(path, "IC_",ARGS[2]) # path to configuration files 

# initialize lattice from metadata
for filename in (readdir(path))
    if occursin(".h5.params", filename)
        f = h5open(string(path, filename), "r")
        break 
    else
        error("No .params file found in $path")
    end 
end
lat = read_lattice(f)
close(f) 

# get momenta 
ks = get_allowed_wavevectors(lat.unit_cell, collect(-L:L)/L)

runMolecularDynamics!(ic_path, dt, 0.0, tmax, lat, ks) # run MD on each rank for all the ICs 
MPI.Barrier(comm)
rank == 0 && @time compute_dynamical_structure_factor(path, dest, Dict("tmax"=>tmax, "dt"=>dt, 
                                                                     "gzz"=>gzz, "theta"=>theta)) # do the averaging on rank 0 