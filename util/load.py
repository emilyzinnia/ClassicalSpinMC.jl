import h5py 
from numpy import array as npa 
import os 
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import pandas as pd
import numpy as np 
import matplotlib.pyplot as pl 

class Params:
    def __init__(self, filename):
        with h5py.File(filename, "r") as f:
            self.sim_params = dict(f.attrs)
            self.S = npa(f["lattice/S"])
            self.shape = npa(f["lattice/size"])
            self.bc = str(f["lattice/bc"].asstr()[...])
            self.N = npa(f["lattice/size"]).prod() * npa(f["unit_cell/basis"]).shape[0] # number of sites 

class SimData:
    def __init__(self, filename):
        self.filename = filename
        with h5py.File(filename, "r") as f:
            paramsfile = f.attrs["paramsfile"].decode('UTF-8')
            paramsfilepath = os.path.dirname(filename) + "/" + os.path.basename(paramsfile)
            self.params = Params(paramsfilepath)
            self.spins = npa(f["spins"])
            self.site_positions = npa(f["site_positions"])
            self.T = f.attrs["T"]
        self.data = {"T": self.T}

    def load_group(self, groupname):
        with h5py.File(self.filename, "r") as f:
            g = f[groupname]
            for key in g.keys():
                self.data[key] = npa(g[key])

    def to_dict(self, keys):
        return dict( (key, self.data[key]) for key in keys )
    
def get_thermal_observables(path):
    '''Reads thermal observables given path with trailing backslash, returns Pandas dataframe'''
    df = []
    for file in os.listdir(path):
        if (".h5" in file) & (".params" not in file):
            data = SimData(path+file)
            data.load_group("observables")
            df.append(data.to_dict(["magnetization", "magnetization_err", 
                                   "specific_heat", "specific_heat_err",
                                   "susceptibility", "susceptibility_err", "T"]))
    return pd.DataFrame(df)


def plot_observable(dat, name, color="blue",**kwargs):
    # sort data by temperature 
    idx = np.argsort(npa(dat["T"]))
    T = npa(dat["T"][idx])
    o = npa(dat[name][idx])
    oerr = npa(dat["{}_err".format(name)][idx])

    fig, ax = pl.subplots()
    vertices = np.block([[T, T[::-1]],
                    [(o+oerr), (o-oerr)[::-1]]]).T
    path_patch = Path(vertices)
    patch = PathPatch(path_patch, facecolor=color, edgecolor='none', alpha=0.5)
    ax.add_patch(patch)
    ax.plot(T, o, color=color, **kwargs)
    return fig, ax 
