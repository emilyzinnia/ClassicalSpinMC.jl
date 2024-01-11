# ClassicalSpinMCJulia

This package is for simulating classical spin systems using Monte Carlo (MC) algorithms. It supports arbitrary lattice constructions up to 3 dimensions with any number of basis sites, and Hamiltonians with a Zeeman field, onsite interactions, and up to quartic interaction terms. 


## Prerequisites 
* `OpenMPI` or `IntelMPI`. The package uses `MPI.jl`, so on a cluster you may need to configure your MPI Julia installation to use the system-provided MPI backend. See the [MPI.jl documentation](https://juliaparallel.org/MPI.jl/stable/configuration/) for details. 
* `HDF5`

## Installation 
1. In your desired installation directory, clone the github repository. 
2. Launch a Julia REPL and type the following command: 

`using Pkg; Pkg.add("$INSTALLATION_PATH/ClassicalSpinMCJulia")`

where `$INSTALLATION_PATH` is the path to the package repository. 

## Usage 
The typical workflow is as follows. 
1. Define lattice, interaction, and Monte Carlo simulation parameters. 
2. Create a `UnitCell(a1,...,an)` where `an` are translation vectors. 
* Add a basis site using `addBasisSite!`. Note that some common geometries (e.g. square and honeycomb) are defined in `bravais.jl`.  
* Add Hamiltonian terms using `addZeemanCoupling!`, `addOnSite!`, `addBilinear!`, `addCubic!` and `addQuartic!`. 
3. Create `Lattice` object using the `UnitCell` object, and by specifying the lattice dimensions, spin magnitude, and boundary conditions (default periodic). 
4. Initialize `MonteCarlo` object. If an output path is specified, a `.h5` file containing the initial spin configuration and a `.h5.params` file containing the simulation metadata will be created. 
5. Perform the desired MC tasks (e.g. simulated annealing or parallel tempering) to thermalize the system to a desired temperature and take measurements. 

### Example: Square lattice Heisenberg model with a field 

We will first do a simulated annealing example on the square lattice. Import the package and set parameters.
```
using ClassicalSpinMC
using LinearAlgebra

L = 4 # lattice size
S = 1.0 # spin magnitude 
J = -1.0 .* collect(I(3)) # interaction matrix 
h = 0.1 # Zeeman field strength
h_c = h .* [0., 0., 1.] # Zeeman field vector 
T = 1e-7 # target temperature 
outpath   = string(pwd(), "/")
```

The Monte Carlo parameters are specified in a dictionary (see the documentation in monte_carlo.jl for details). For simulated annealing, we need to define the number of thermalization sweeps and number of overrelaxation sweeps per Metropolis sweep. 

```
mcparams  = Dict( "t_thermalization" => Int(1e5),     
                  "t_deterministic" => Int(1e6),
                  "overrelaxation"   => 10      )     
```

Next, generate a unit cell object and add Hamiltonian terms. 

```
UC = Square()  
addBilinear!(S, 1, 1, J, (1, 0)) #x+
addBilinear!(S, 1, 1, J, (-1, 0)) #x-
addBilinear!(S, 1, 1, J, (0, 1 )) #y+
addBilinear!(S, 1, 1, J, (0, -1 )) #y-
addZeemanCoupling!(S, 1, h_c)
```

Next, we create the lattice object with periodic boundary conditions (by default). 

```
lat = Lattice( (L,L), UC, S, bc="periodic") 
```

We then initialize and construct the MC object

```
mc = MonteCarlo(T, lat, mcparams, outpath=outpath)
```

Because we specified an output path, this line will create `configuration.h5.params` and `configuration_0.h5` files. 

Finally, we perform simulated annealing with deterministic updates (used at very low temperatures when the metropolis acceptance rate is almost nonexistent).

```
simulated_annealing!(mc, x ->1.0*0.9^x, 1.0)
deterministic_updates!(mc)
```

The current spin configuration is stored in `mc.lattice.spins`, and can be outputted to `configuration_0.h5` using 

```
write_MC_checkpoint(mc)
```
