using Base.Iterators: float
using Base: Real, Number
using MPI 
using HDF5

using Pkg
Pkg.activate("ClassicalSpinMC")
using ClassicalSpinMC

include("honeycomb.jl")

# simulation parameters 
tstep = 0.01
tmin = 0.0
tmax = 40.0
dir = [1/sqrt(3),  1/3]

# initialize MPI
MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD
commSize = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

path = string(ARGS[1], "IC_0/")
dest = string(ARGS[1], "configuration_0.h5")
folder = ARGS[1]

rank == 0 && println("*****************Running for $folder*****************")

# initialize lattice 
f = h5open(dest, "r")
U = Honeycomb()
size = read(attributes(f)["shape"])
S = read(attributes(f)["S"])
h = read(attributes(f)["S"])
lat = lattice(size, U; field=h)
ks = get_k_path(U, dir, size[1])

# do MD simulation 
runMolecularDynamics!(path, dir, tstep, tmin, tmax, lat, ks)
MPI.Barrier(comm)
rank == 0 && @time compute_dynamic_structure_factor(path, dest, Dict("tmax"=>tmax, "tstep"=>tstep))


