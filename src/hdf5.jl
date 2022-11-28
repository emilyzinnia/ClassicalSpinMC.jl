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

# dims is the number of measurements that will be taken 
function initialize_hdf5(filename, mc)
    file = h5open(filename, "w")
    # dump all input parameters 
    dump_attributes_hdf5!(file, mc.input_parameters)
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
    g["magnetization"] = mean(mc.observables.magnetization)
    g["magnetization_err"] = std_error(mc.observables.magnetization)
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