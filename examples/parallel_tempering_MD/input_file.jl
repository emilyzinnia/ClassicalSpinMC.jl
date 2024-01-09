
#------------------------------------------
# constants
#------------------------------------------
k_B = 1/11.6 # meV/K
mu_B = 0.67*k_B # K/T to meV/T

#------------------------------------------
# local z axis for each basis site 
#------------------------------------------
z1 = [1, 1, 1]/sqrt(3)
z2 = [1,-1,-1]/sqrt(3)
z3 = [-1,1,-1]/sqrt(3)
z4 = [-1,-1,1]/sqrt(3)

#------------------------------------------
# set MC parameters
#------------------------------------------
t_thermalization= Int(1e6)
t_measurement= Int(1e6)
probe_rate=2000
swap_rate=50
overrelaxation=10
report_interval = Int(1e4)
checkpoint_rate=1000

#------------------------------------------
# set lattice parameters 
#------------------------------------------
L = 8
S = 1/2

#------------------------------------------
# set Zeeman coupling parameters
#------------------------------------------
gxx = 0.0
gzz = 2.18
gyy = 0.0
# h = [1, 1, 1]/sqrt(3)
h = [1, 0, 0]
h1 = (h'*z1) * [gxx, gyy, gzz]
h2 = (h'*z2) * [gxx, gyy, gzz]
h3 = (h'*z3) * [gxx, gyy, gzz]
h4 = (h'*z4) * [gxx, gyy, gzz]

#------------------------------------------
# set interaction parameters in meV
#------------------------------------------
Jxx = Jzz = 0.043
Jyy = 0.065

#------------------------------------------
# set temperatures in units of meV
#------------------------------------------
Tmin = 0.09 *k_B
Tmax = 14 *k_B
