import numpy as np
from numpy import array as npa
from matplotlib.tri import Triangulation
from matplotlib.patches import Polygon
from sympy import Point, Segment
from load import * 

# Neutron scattering structure factor definitions
def structure_factor(spins, qs, basis=np.eye(3)):
    # if any of the momenta are zero, multiply by tiny number to account for divergence at q=0
    cond = (qs[:,0] == 0) & (qs[:,1] == 0) & (qs[:,2] == 0)
    qs[cond] = np.ones(3) * 1e-8
    qsquared = qs[:, 0]**2+qs[:, 1]**2+qs[:, 2]**2

    SF= np.zeros(spins.shape[:-1])
    for a in range(3):
        for b in range(3):
            projector = np.dot(basis[:,a], basis[:,b]) - (
                np.einsum("ij,j",qs, basis[:,a]) * np.einsum("ij,j",qs, basis[:,b])) / qsquared
            SF += projector * spins[...,a+b+(2*a)] 
    return SF

def reflectx(point):
    return npa([-point[0], point[1]])

def reflecty(point):
    return npa([point[0], -point[1]])

def get_intersections(p1, p2):
    '''Finds perpendicular bisectors of two lines and returns intersections'''
    l1 = Segment( Point(0,0), Point(*p1) ).perpendicular_bisector()
    l2 = Segment( Point(0,0), Point(*p2) ).perpendicular_bisector()
    return float(l1.intersection(l2)[0][0]), float(l1.intersection(l2)[0][1])

def reciprocal(*args):
    if len(args) == 2: 
        a1, a2 = args 
        mag = 2*np.pi  / (a1[0]*a2[1] - a1[1]*a2[0])
        b1 = mag * npa([a2[1], -a2[0]])
        b2 = mag * npa([-a1[1], a1[0]])
        return (b1, b2)
    elif len(args) == 3:
        a1, a2, a3 = args 
        mag = 2*np.pi  / np.dot(a1, np.cross(a2, a3)) 
        b1 = mag * np.cross(a2, a3)
        b2 = mag * np.cross(a3, a1)
        b3 = mag * np.cross(a1, a2)
        return (b1, b2, b3) 
    else:
        raise ValueError("Expected either 2 or 3 arguments")

def draw_FBZ_2D(*points):
    # get a set of all reflections 
    set = []
    for i, p in enumerate(points):
        set.append( p )
        set.append( reflectx(reflecty(np.copy(p))))
        set.append( reflectx(np.copy(p)) )
        set.append( reflecty(np.copy(p)) ) 
    set = np.unique(npa(set), axis=0 ) 

    # sort in clockwise order by converting to polar coordinates
    theta = np.rad2deg(np.arctan2( set[:,1], set[:,0]) )
    idx = np.argsort(theta)
    set = set[idx,:]

    # get vertices of polygon
    pairs = np.roll(set, -1, axis=0)
    # get_intersections(set[0,:], pairs[0])
    vertices = npa([ get_intersections(p, pairs[i]) for i, p in enumerate(set)]) 
    return vertices

def plot_SSF(data, figobjects, R=np.eye(3), **kwargs):
    fig, ax = figobjects
    # get data 
    wf = data.data["SSF_momentum"] 
    Suv = data.data["SSF"]

    # compute neutron structure factor 
    tot = structure_factor(Suv, np.c_[wf, np.zeros(wf.shape[0])], basis=R) 

    # triangulation for plotting 
    lim = (-2*np.pi <= wf[:,0]) & (wf[:,0]<= 2*np.pi) & (-2*np.pi <= wf[:,1]) & (wf[:,1]<= 2*np.pi)
    triMesh = Triangulation(wf[:,0][lim], wf[:,1][lim])
    cs = ax.tripcolor(triMesh, tot[lim], **kwargs)
    cb = fig.colorbar(cs, ax=ax, shrink=0.7, use_gridspec=True)

    # plot FBZ 
    bz = draw_FBZ_2D(*reciprocal(*data.params.lattice_vectors))
    ax.add_patch(Polygon(bz, closed=True, facecolor='none', edgecolor="white",linewidth=1))

    r = np.arange(-2, 3)
    labels = [r'${}\pi$'.format(num) if num != 0 else "0" for num in r]

    ax.set_aspect("equal")
    ax.set_xlim([-2*np.pi, 2*np.pi])
    ax.set_ylim([-2*np.pi, 2*np.pi])
    ax.set_facecolor("black")
    ax.set_xticks(r*np.pi)
    ax.set_yticks(r*np.pi)
    ax.set_xticklabels( labels)
    ax.set_yticklabels( labels)