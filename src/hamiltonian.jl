using Einsum 

function get_local_field(lattice::Lattice, point::Int64)
    @inbounds js = get_bilinear_sites(lattice, point)
    @inbounds Js = get_bilinear_matrices(lattice, point)
    @inbounds cs = get_cubic_sites(lattice, point)
    @inbounds rs = get_quartic_sites(lattice, point)
    @inbounds h = get_field(lattice, point)
    @inbounds o = get_onsite(lattice, point)
    @inbounds Rs = get_quartic_tensors(lattice, point)
    @inbounds Cs = get_cubic_tensors(lattice, point)

    # sum over all interactions
    Hx = 0.0
    Hy = 0.0
    Hz = 0.0

    # on-site interaction
    @inbounds six, siy, siz = get_spin(lattice.spins, point) 
    @inbounds Hx += 2 * ( o.m11 * six + o.m12 * siy + o.m13 * siz)
    @inbounds Hy += 2 * ( o.m21 * six + o.m22 * siy + o.m23 * siz)
    @inbounds Hz += 2 * ( o.m31 * six + o.m32 * siy + o.m33 * siz)

    # bilinear interaction 
    for n in eachindex(js)
        J = Js[n]
        @inbounds sjx, sjy, sjz = get_spin(lattice.spins, js[n])
        @inbounds Hx += J.m11 * sjx + J.m12 * sjy + J.m13 * sjz 
        @inbounds Hy += J.m21 * sjx + J.m22 * sjy + J.m23 * sjz 
        @inbounds Hz += J.m31 * sjx + J.m32 * sjy + J.m33 * sjz 
    end

    # cubic interaction 
    for n in eachindex(cs)
        C = Cs[n]
        j, k = cs[n]
        @inbounds sj = get_spin(lattice.spins, j)
        @inbounds sk = get_spin(lattice.spins, k)

        @einsum Hx += C[1, a, b] * sj[a] * sk[b]
        @einsum Hy += C[2, a, b] * sj[a] * sk[b]
        @einsum Hz += C[3, a, b] * sj[a] * sk[b]
    end

    # quadratic interaction 
    for n in eachindex(rs)
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
    E_cubic = 0.0
    E_quartic = 0.0
    E_zeeman = 0.0
    E_onsite = 0.0
    
    for point in 1:lattice.size 
        @inbounds js = get_bilinear_sites(lattice, point)
        @inbounds Js = get_bilinear_matrices(lattice, point)
        @inbounds cs = get_cubic_sites(lattice, point)
        @inbounds rs = get_quartic_sites(lattice, point)
        @inbounds h = get_field(lattice, point)
        @inbounds o = get_onsite(lattice, point)
        @inbounds Rs = get_quartic_tensors(lattice, point)
        @inbounds Cs = get_cubic_tensors(lattice, point)

        s = get_spin(lattice.spins, point)
        E_zeeman -= (s[1] * h[1] + s[2] * h[2] + s[3] * h[3]) 
        E_onsite += s[1] * (o.m11 * s[1] + o.m12 * s[2] + o.m13 * s[3]) + 
                    s[2] * (o.m21 * s[1] + o.m22 * s[2] + o.m23 * s[3]) + 
                    s[3] * (o.m31 * s[1] + o.m32 * s[2] + o.m33 * s[3])  

        for n in eachindex(js)
            J = Js[n]
            @inbounds sj = get_spin(lattice.spins, js[n])
            @inbounds E_bilinear +=  s[1] * (J.m11 * sj[1] + J.m12 * sj[2] + J.m13 * sj[3]) + 
                            s[2] * (J.m21 * sj[1] + J.m22 * sj[2] + J.m23 * sj[3]) + 
                            s[3] * (J.m31 * sj[1] + J.m32 * sj[2] + J.m33 * sj[3]) 
        end

        # cubic term
        for n in eachindex(cs)
            C = Cs[n]
            j, k = cs[n]
            @inbounds sj = get_spin(lattice.spins, j)
            @inbounds sk = get_spin(lattice.spins, k)
            @einsum E_cubic += C[a, b, c] * s[a] * sj[b] * sk[c]
        end

        # quartic term 
        for n in eachindex(rs)
            R = Rs[n]
            j, k, l = rs[n]
            @inbounds sj = get_spin(lattice.spins, j)
            @inbounds sk = get_spin(lattice.spins, k)
            @inbounds sl = get_spin(lattice.spins, l)
            @einsum E_quartic += R[a, b, c, d] * s[a] * sj[b] * sk[c] * sl[d]
        end
        
    end
    return E_bilinear/2 + E_cubic/3 + E_quartic/4 + E_zeeman + E_onsite 
end

function energy_density(lattice::Lattice)
    return total_energy(lattice) / lattice.size
end

# calculates the energy at one site
function energy(lattice::Lattice, point::Int64)::Float64
    @inbounds js = get_bilinear_sites(lattice, point)
    @inbounds Js = get_bilinear_matrices(lattice, point)
    @inbounds cs = get_cubic_sites(lattice, point)
    @inbounds rs = get_quartic_sites(lattice, point)
    @inbounds h = get_field(lattice, point)
    @inbounds Rs = get_quartic_tensors(lattice, point)
    @inbounds Cs = get_cubic_tensors(lattice, point)
    @inbounds o = get_onsite(lattice, point)

    s = get_spin(lattice.spins, point)

    # sum over all interactions
    E = 0.0

    # onsite term 
    E += s[1] * (o.m11 * s[1] + o.m12 * s[2] + o.m13 * s[3]) + 
         s[2] * (o.m21 * s[1] + o.m22 * s[2] + o.m23 * s[3]) + 
         s[3] * (o.m31 * s[1] + o.m32 * s[2] + o.m33 * s[3])  
    
    # bilinear term
    for n in eachindex(js)
        J = Js[n]
        @inbounds sj = get_spin(lattice.spins, js[n])
        @inbounds E +=  s[1] * (J.m11 * sj[1] + J.m12 * sj[2] + J.m13 * sj[3]) + 
                        s[2] * (J.m21 * sj[1] + J.m22 * sj[2] + J.m23 * sj[3]) + 
                        s[3] * (J.m31 * sj[1] + J.m32 * sj[2] + J.m33 * sj[3]) 
    end

    # cubic term 
    for n in eachindex(cs)
        C = Cs[n]
        j, k = cs[n]
        @inbounds sj = get_spin(lattice.spins, j)
        @inbounds sk = get_spin(lattice.spins, k)
        @einsum E += C[a, b, c] * s[a] * sj[b] * sk[c] 
    end

    # quartic term 
    for n in eachindex(rs)
        R = Rs[n]
        j, k, l = rs[n]
        @inbounds sj = get_spin(lattice.spins, j)
        @inbounds sk = get_spin(lattice.spins, k)
        @inbounds sl = get_spin(lattice.spins, l)
        @einsum E += R[a, b, c, d] * s[a] * sj[b] * sk[c] * sl[d]
    end
    return E -(s[1] * h[1] + s[2] * h[2] + s[3] * h[3]) 
end