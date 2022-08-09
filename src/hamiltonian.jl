using Einsum 

function get_local_field(lattice::Lattice, point::Int64)
    @inbounds js = get_interaction_sites(lattice, point)
    @inbounds Js = get_interaction_matrices(lattice, point)
    @inbounds rs = get_ring_exchange_sites(lattice, point)
    Rs = lattice.ring_exchange_tensors

    h = lattice.field
    # sum over all interactions
    Hx = 0.0
    Hy = 0.0
    Hz = 0.0
    for n=1:length(js)
        J = Js[n]
        @inbounds sjx, sjy, sjz = get_spin(lattice.spins, js[n])
        @inbounds Hx += J.m11 * sjx + J.m12 * sjy + J.m13 * sjz 
        @inbounds Hy += J.m21 * sjx + J.m22 * sjy + J.m23 * sjz 
        @inbounds Hz += J.m31 * sjx + J.m32 * sjy + J.m33 * sjz 
    end

    for n=1:length(rs)
        R = Rs[n]
        j, k, l = rs[n]
        @inbounds sj = get_spin(lattice.spins, j)
        @inbounds sk = get_spin(lattice.spins, k)
        @inbounds sl = get_spin(lattice.spins, l)

        @einsum Hx += R[1, a, b, c] * sj[a] * sk[b] * sl[c]
        @einsum Hy += R[2, a, b, c] * sj[a] * sk[b] * sl[c]
        @einsum Hz += R[3, a, b, c] * sj[a] * sk[b] * sl[c]
    end
    return (Hx-h[1], Hy-h[2], Hz-h[3])
end


function total_energy(lattice::Lattice)
    # sum over all nearest neighbours
    E_bilinear = 0.0
    E_ring = 0.0
    E_zeeman = 0.0
    h = lattice.field 
    Rs = lattice.ring_exchange_tensors
    
    for point in 1:lattice.size 
        @inbounds js = get_interaction_sites(lattice, point)
        @inbounds Js = get_interaction_matrices(lattice, point)
        @inbounds rs = get_ring_exchange_sites(lattice, point)
        
        s = get_spin(lattice.spins, point)
        E_zeeman -= (s[1] * h[1] + s[2] * h[2] + s[3] * h[3]) 
        for n=1:length(js)
            J = Js[n]
            @inbounds sj = get_spin(lattice.spins, js[n])
            @inbounds E_bilinear +=  s[1] * (J.m11 * sj[1] + J.m12 * sj[2] + J.m13 * sj[3]) + 
                            s[2] * (J.m21 * sj[1] + J.m22 * sj[2] + J.m23 * sj[3]) + 
                            s[3] * (J.m31 * sj[1] + J.m32 * sj[2] + J.m33 * sj[3]) 
        end

            # ring exchange term 
        for n=1:length(rs)
            R = Rs[n]
            j, k, l = rs[n]
            @inbounds sj = get_spin(lattice.spins, j)
            @inbounds sk = get_spin(lattice.spins, k)
            @inbounds sl = get_spin(lattice.spins, l)
            @einsum E_ring += R[a, b, c, d] * s[a] * sj[b] * sk[c] * sl[d]
        end
        
    end
    return E_bilinear/2 + E_ring/4 + E_zeeman
end

function energy_density(lattice::Lattice)
    return total_energy(lattice) / lattice.size
end

# calculates the energy at one site
function energy(lattice::Lattice, point::Int64)::Float64
    @inbounds js = get_interaction_sites(lattice, point)
    @inbounds Js = get_interaction_matrices(lattice, point)
    @inbounds rs = get_ring_exchange_sites(lattice, point)
    @inbounds Rs = lattice.ring_exchange_tensors

    s = get_spin(lattice.spins, point)
    h = lattice.field 

    # sum over all nearest neighbours
    E = 0.0
    for n=1:length(js)
        J = Js[n]
        @inbounds sj = get_spin(lattice.spins, js[n])
        @inbounds E +=  s[1] * (J.m11 * sj[1] + J.m12 * sj[2] + J.m13 * sj[3]) + 
                        s[2] * (J.m21 * sj[1] + J.m22 * sj[2] + J.m23 * sj[3]) + 
                        s[3] * (J.m31 * sj[1] + J.m32 * sj[2] + J.m33 * sj[3]) 
    end

    # ring exchange term 
    for n=1:length(rs)
        R = Rs[n]
        j, k, l = rs[n]
        @inbounds sj = get_spin(lattice.spins, j)
        @inbounds sk = get_spin(lattice.spins, k)
        @inbounds sl = get_spin(lattice.spins, l)

        @einsum E += R[a, b, c, d] * s[a] * sj[b] * sk[c] * sl[d]
    end
    return E -(s[1] * h[1] + s[2] * h[2] + s[3] * h[3]) 
end