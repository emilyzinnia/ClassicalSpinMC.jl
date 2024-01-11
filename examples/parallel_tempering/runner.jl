using ClassicalSpinMC
using MPI

include("pyrochlore.jl")
include("input_file.jl")

# initialize MPI
MPI.Initialized() || MPI.Init()
commSize = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# get temperature on rank 
# create equal logarithmically spaced temperatures and get temperature on current rank 
temp = exp10.(range(log10(Tmin), stop=log10(Tmax), length=commSize) ) 
T = temp[rank+1] # target temperature for rank

# read command line arguments 
path = length(ARGS) == 0 ? string(pwd(),"/") : ARGS[1] 
B = 0.0

# create unit cell 
P = Pyrochlore()

# add Hamiltonian terms
interactions = Dict("Jxx"=>Jxx, "Jyy"=>Jyy, "Jzz"=>Jzz)
addInteractionsLocal!(P, interactions) # bilinear spin term 
addZeemanCoupling!(P, 1, h1*B*mu_B) # add zeeman coupling to each basis site
addZeemanCoupling!(P, 2, h2*B*mu_B)
addZeemanCoupling!(P, 3, h3*B*mu_B)
addZeemanCoupling!(P, 4, h4*B*mu_B)

# generate lattice
lat = Lattice( (L,L,L), P, S) 
params = Dict("t_thermalization"=>t_thermalization, "t_measurement"=>t_measurement, 
                "probe_rate"=>probe_rate, "swap_rate"=>swap_rate, "overrelaxation_rate"=>overrelaxation, 
                "report_interval"=>report_interval, "checkpoint_rate"=>checkpoint_rate)

# create MC object 
mc = MonteCarlo(T, lat, params)

# perform MC tasks 
parallel_tempering!(mc, [0]) # output measurements on rank 0
