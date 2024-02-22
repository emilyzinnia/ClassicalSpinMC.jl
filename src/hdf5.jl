using HDF5

"""
Helper function to dump attributes from compound datatypes to hdf5. 
"""
function dump_attributes_hdf5!(fid, obj)
    input_ = fieldnames(typeof(obj))
    # dump all input parameters 
    for param in input_
        write_attribute(fid, String(param), getfield(obj, param))
    end
end

"""
Helper function to dump attributes from dictionaries to hdf5. 
"""
function dump_attributes_hdf5!(fid, dict::Dict{String,<:Any})
    # dump all input parameters 
    for (key, value) in dict 
        write_attribute(fid, key, value)
    end
end

"""
Write user-specified dictionary to h5 file.
"""
function write_attributes(filename::String, dict::Dict{String,<:Any})
    f = h5open(filename, "r+")
    dump_attributes_hdf5!(f, dict)
    close(f)
end

"""
Dump UnitCell object metadata into h5 file. 
"""
function dump_unit_cell!(fid, uc::UnitCell)
    g = create_group(fid, "unit_cell")
    # dump unit cell fields to h5 file 
    g["lattice_vectors"] = tuple_to_matrix(uc.lattice_vectors)
    g["basis"] = reduce(vcat,transpose.(uc.basis))

    # zeeman field 
    h = create_group(g, "field")
    for element in uc.field 
        basis, vec = element 
        h[string(basis)] = vec 
    end 

    # onsite 
    o = create_group(g, "onsite")
    for element in uc.onsite
        b1, mat = element 
        o[string(b1)] = IMToMatrix(mat) # convert interaction matrix to regular matrix 
    end

    # bilinear 
    b = create_group(g, "bilinear")
    for element in uc.bilinear 
        b1, b2, mat, offset = element 
        b[string("($b1,$b2),$offset")] = IMToMatrix(mat)
    end 

    # cubic 
    c = create_group(g, "cubic")
    for element in uc.cubic 
        b1, b2, b3, cub, o2, o3 = element 
        c[string("($b1,$b2,$b3),$o2,$o3")] = cub 
    end 

    # quartic 
    q = create_group(g, "quartic")
    for element in uc.quartic 
        b1, b2, b3, b4, quar, o2, o3, o4 = element 
        q[string("($b1,$b2,$b3,$b4),$o2,$o3,$o4")] = quar
    end 
end

"""
Reads unit_cell group from hdf5.params file and returns unit cell object.
"""
function read_unit_cell(fid)
    g = fid["unit_cell"]
    UC = UnitCell(eachcol(read(g["lattice_vectors"]))...)

    # basis 
    basis = tuple(eachrow(read(g["basis"]))...)
    for site in basis 
        addBasisSite!(UC, collect(site))
    end

    # Zeeman field 
    for key in keys(g["field"])
        addZeemanCoupling!(UC, parse(Int64, key), read(g["field/$key"]))
    end 

    # onsite 
    for key in keys(g["onsite"])
        addOnSite!(UC, parse(Int64, key), read(g["onsite/$key"]))
    end 

    # bilinear 
    for key in keys(g["bilinear"])
        bs, offset = eval(Meta.parse(key))
        addBilinear!(UC, bs..., read(g["bilinear/$key"]), offset)
    end 

    # cubic
    for key in keys(g["cubic"])
        bs, o2, o3 = eval(Meta.parse(key))
        addCubic!(UC, bs..., read(g["cubic/$key"]), o2, o3)
    end 

    # quartic
    for key in keys(g["quartic"])
        bs, o2, o3, o4 = eval(Meta.parse(key))
        addQuartic!(UC, bs..., read(g["quartic/$key"]), o2, o3, o4)
    end 

    return UC 
end

"""
Dump MonteCarlo object metadata into .params file. 
"""
function dump_metadata!(fid, mc)
    # dump unit cell metadata 
    dump_unit_cell!(fid, mc.lattice.unit_cell)

    # dump lattice data 
    l = create_group(fid, "lattice")
    l["size"] = collect(mc.lattice.shape)
    l["S"] = mc.lattice.S
    l["bc"] = mc.lattice.bc

    # dump MC simulation parameters 
    dump_attributes_hdf5!(fid, mc.parameters)
end 

"""
Create .params file at path specified by MonteCarlo.outpath.
"""
function create_params_file(mc, filename)
    f = h5open(filename, "w")
    dump_metadata!(f, mc)
    close(f)
    return filename
end

"""
Reads lattice group from h5.params file and returns Lattice object.
"""
function read_lattice(fid)::Lattice
    l = fid["lattice"]
    size = tuple(read(l["size"])...)
    uc = read_unit_cell(fid)
    S = read(l["S"])
    bc = read(l["bc"])
    return Lattice(size, uc, S, bc=bc)
end 

"""
Create configuration file at specified temperature for output. 
"""
function initialize_hdf5(mc, paramsfile::String)
    file = h5open(mc.outpath, "w")
    write_attribute(file, "T", mc.T)
    write_attribute(file, "paramsfile", paramsfile)
    file["spins"] = mc.lattice.spins
    file["site_positions"] = mc.lattice.site_positions
    close(file)
end

"""
Update spin configuration in configuration file. 
"""
function write_MC_checkpoint(mc)
    file = h5open(mc.outpath, "r+")
    file["spins"][:,:] = mc.lattice.spins #overwrite current spins
    close(file)
end

"""
Write spins of current spin configuration to .h5 file.
"""
function write_initial_configuration(filename::String, mc)
    file = h5open(filename, "w")
    write_attribute(file, "T", mc.T)
    params_ = h5open(mc.outpath, "r")
    paramsfile = read(attributes(params_)["paramsfile"])
    write_attribute(file, "paramsfile", paramsfile)
    file["spins"] = mc.lattice.spins 
    close(file)
    close(params_)
end

"""
Update configuration file with final observables. Outputs:
- specific heat and error 
- magnetization and error 
- magnetic susceptibility and error 
- total energy and error 
- roundtrip marker and error (for tracking parallel tempering success purposes)
"""
function write_final_observables(mc) 
    file = h5open(mc.outpath, "r+")
    file["spins"][:,:] = mc.lattice.spins  # dump final spins 
    heat, dheat = specific_heat(mc)
    chi, dchi = susceptibility(mc)
    
    if haskey(file, "observables")
        g = file["observables"]
        for key in keys(g)
            delete_object(g, key)
        end
        println("Overwriting observables in $filename")
    else
        g = create_group(file, "observables")
    end

    g["specific_heat"] = heat 
    g["specific_heat_err"] = abs(dheat/heat)
    g["susceptibility"] = chi
    g["susceptibility_err"] = abs(dchi/chi)
    g["magnetization"] = mean(mc.observables.magnetization, 1)
    g["magnetization_err"] = std_error(mc.observables.magnetization, 1)
    g["energy"] = mean(mc.observables.energy,1)
    g["energy_err"] = std_error(mc.observables.energy,1)
    close(file)
end

"""
Overwrites existing keys in h5 file.
"""
function overwrite_keys!(fid, dict)
    for key in keys(dict)
        if haskey(fid, key)
            delete_object(fid, key)
            println("Overwriting parameter $key")
        end
        fid[key] = dict[key]
    end
end

function attributes_to_dict(file)
    attr = attributes(file)
    dict = Dict()
    for key in keys(attr)
        push!(dict, key => read(attr[key]))
    end
    return dict 
end

"""
Updates spins in lattice object with spin configuration from .h5 file.
"""
function read_spin_configuration!(lat::Lattice, filename::String)
    file = h5open(filename, "r")
    lat.spins[:,:] = read(file["spins"])
    close(file)
end