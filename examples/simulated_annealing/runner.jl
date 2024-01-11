using ClassicalSpinMC
include("honeycomb.jl")

#------------------------------------------
# set lattice parameters 
#------------------------------------------
L = 4
S = 1.0

#------------------------------------------
# set interaction parameters 
#------------------------------------------
K = -1.0
h = 0.1
h_vec = h*[-1,1,0]/sqrt(2)
inparams = Dict("K"=>K, "h"=>h)  # dictionary of human readable input parameters for output 

#------------------------------------------
# set MC parameters 
#------------------------------------------
mcparams  = Dict( "t_thermalization" => Int(1e4),
                  "t_deterministic" => Int(1e6),
                  "overrelaxation_rate"   => 10      )
outpath   = string(pwd(), "/")

# target temperature
T = 1e-7

# generate honeycomb unit cell
H = Honeycomb()

# add Hamiltonian terms
addInteractionsKitaev!(H, Dict("K"=>K))
addZeemanCoupling!(H, 1, h_vec)
addZeemanCoupling!(H, 2, h_vec)

# create lattice 
lat = Lattice( (L,L), H, S) 

# initialize MC struct
mc = MonteCarlo(T, lat, mcparams, outpath=outpath, outprefix="configuration", inparams=inparams)

# perform MC tasks 
simulated_annealing!(mc, x ->1.0*0.9^x, 1.0)
deterministic_updates!(mc)

# write to file 
write_MC_checkpoint(mc)
