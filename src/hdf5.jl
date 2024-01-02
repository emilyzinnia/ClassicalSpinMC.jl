using HDF5

#helper function to dump attributes from compound datatypes to hdf5
function dump_attributes_hdf5!(fid, obj)
    input_ = fieldnames(typeof(obj))
    # dump all input parameters 
    for param in input_
        write_attribute(fid, String(param), getfield(obj, param))
    end
end

function dump_attributes_hdf5!(fid, dict::Dict{String,Float64})
    # dump all input parameters 
    for (key, value) in dict 
        write_attribute(fid, key, value)
    end
end

# converts tuples of vectors or tuples of other tuples to 2D matrix 
function tuple_to_matrix(tuple_data)
    return reshape(collect(Iterators.flatten(tuple_data)), length(tuple_data), length(tuple_data[1]))
end

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
        o[string(b1)] = Matrix(mat) # convert interaction matrix to regular matrix 
    end

    # bilinear 
    b = create_group(g, "bilinear")
    for element in uc.bilinear 
        b1, b2, mat, offset = element 
        b[string("($b1,$b2),$offset")] = Matrix(mat)
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

# reads unit_cell group from hdf5.params file and returns unit cell object 
function read_unit_cell(fid)
    g = fid["unit_cell"]
    UC = UnitCell(eachrow(read(g["lattice_vectors"]))...)

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

# dims is the number of measurements that will be taken 
function initialize_hdf5(filename, mc)
    file = h5open(filename, "w")
    dump_attributes_hdf5!(file, mc.parameters)
    write_attribute(file, "T", mc.T)
    write_attribute(file, "shape", collect(mc.lattice.shape))
    file["spins"] = mc.lattice.spins
    file["site_positions"] = mc.lattice.site_positions
    close(file)
end

function write_MC_checkpoint!(filename::String, mc)
    file = h5open(filename, "r+")
    file["spins"][:,:] = mc.lattice.spins #overwrite current spins
    close(file)
end

function write_initial_configuration!(filename::String, mc)
    file = h5open(filename, "w")
    dump_attributes_hdf5!(file, mc.parameters)
    write_attribute(file, "shape", collect(mc.lattice.shape[2:end]))
    write_attribute(file, "T", mc.T)
    file["spins"] = mc.lattice.spins 
    close(file)
end

function write_final_observables!(filename::String, mc) 
    file = h5open(filename, "r+")
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
    g["roundtripMarker"] = mean(mc.observables.roundtripMarker)
    g["energy_timeseries"] = collect(mc.observables.energyTimeSeries)
    g["roundtripMarker_err"] = std_error(mc.observables.roundtripMarker)
    close(file)
end

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

function read_spin_configuration!(filename::String, lat::Lattice)
    file = h5open(filename, "r")
    lat.spins[:,:] = read(file["spins"])
    close(file)
end