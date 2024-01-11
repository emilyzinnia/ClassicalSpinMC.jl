------------------------------------------------------------
pyrochlore.jl
------------------------------------------------------------
- function wrappers for adding the interaction matrices between sites in a unit cell in the global or local frames

------------------------------------------------------------
input_file.jl
------------------------------------------------------------
- input file containing global variables to be used in the script
- note that this file isn't mandatory, you could of course define all these variables in the "runner.jl" script

------------------------------------------------------------
runner.jl
------------------------------------------------------------
- run finite temperature MC using the parallel tempering algorithm
- first thermalizes the system for t_sweep MC sweeps, then takes measurements for t_measurement MC sweeps
- produces one temperature per CPU and swaps configurations across CPUs using MPI
- generates one spin configuration_X.h5 file per temperature, and also N measurement files (optional) for specified ranks to be used as initial conditions for MD

to run from command line:
mpiexecjl -n $NTASKS julia runner.jl $OUTPUT_PATH

(you can use your favourite MPI executable, but I just use the julia mpiexec wrapper)
