function Triangular()::UnitCell
    # create unit cell
    a1 = [1.0, 0.0]
    a2 = cos(pi/3) * [1,0] + sin(pi/3) * [0, 1]

    UC = UnitCell(a1, a2)
    return UC
end 

function Square()::UnitCell
    # create unit cell
    a1 = [1.0, 0.0]
    a2 = [0.0, 1.0]
    UC = UnitCell(a1, a2)
    return UC
end 

function FCC()::UnitCell
    # create unit cell
    a1 = 0.5 * [0,1,1]
    a2 = 0.5 * [1,0,1]
    a3 = 0.5 * [1,1,0]
    UC = UnitCell(a1, a2, a3)
    return UC
end 

function Pyrochlore()::UnitCell
    basis = ([1,1,1]/8, [1,-1,-1]/8, [-1,1,-1]/8, [-1,-1,1]/8) 
    UC = FCC()
    for site in basis 
        addBasisSite!(UC, site )
    end
    return UC 
end

function BreathingPyrochlore(a::Float64=1.01)::UnitCell
    basis = a .* ([1,1,1]/8, [1,-1,-1]/8, [-1,1,-1]/8, [-1,-1,1]/8) 
    UC = FCC()
    for site in basis 
        addBasisSite!(UC, site )
    end
    return UC 
end

function Honeycomb()::UnitCell
    # create unit cell
    UC = Triangular()
    addBasisSite!(UC, [0.0, 0.0])
    addBasisSite!(UC, [0.0, 1.0]/sqrt(3))
    return UC
end 
