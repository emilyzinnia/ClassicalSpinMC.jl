"""
Add interactions to UnitCell object in the global cartesian frame. 
"""
function addInteractionsGlobal!(U::UnitCell, dict::Dict{String, Float64})
      allowed_keys = ["Jxx", "Jyy", "Jzz"]
      for key in allowed_keys
            if !(key in keys(dict))
                  dict[key] = 0.0
            end
      end
      Jx = dict["Jxx"]
      Jy = dict["Jyy"]
      Jz = dict["Jzz"]

      J1 = 1/6 * (-2Jx + 2Jz)
      J2 = 1/3 * (-2Jx - Jz)
      J3 = 1/6 * (Jx -3Jy +2Jz)
      J4 = 1/6 * (-Jx -3Jy -2Jz)
  
      J12 = [ J1 J2 -J1
              J3 -J1 J4
              -J4 -J1 -J3 ]
      J13 = [ -J1 J1 J2 
              J4 J3 -J1 
              -J3 -J4 -J1]
      J14 = [ -J1 -J1 -J2 
              J4 -J3 J1 
              -J3 J4 J1 ]
      J23 = [ -J3 -J4 -J1 
              J1 -J1 -J2 
              -J4 -J3 J1]
      J24 = [ -J3 J4 J1 
              J1 J1 J2
              -J4 J3 -J1 ]
      J34 = [ -J4 J3 -J1
              -J3 J4 J1
              J1 J1 J2 ]

      # A neighbours 
      addBilinear!(U, 1, 2, J12, (0, 0, 0)) 
      addBilinear!(U, 1, 3, J13, (0, 0, 0)) 
      addBilinear!(U, 1, 4, J14, (0, 0, 0)) 
      addBilinear!(U, 2, 3, J23, (0, 0, 0)) 
      addBilinear!(U, 2, 4, J24, (0, 0, 0)) 
      addBilinear!(U, 3, 4, J34, (0, 0, 0)) 

      # B neighbours
      addBilinear!(U, 1, 2, J12, (1, 0, 0)) 
      addBilinear!(U, 1, 3, J13, (0, 1, 0)) 
      addBilinear!(U, 1, 4, J14, (0, 0, 1)) 
      addBilinear!(U, 2, 3, J23, (-1, 1, 0)) 
      addBilinear!(U, 2, 4, J24, (-1, 0, 1)) 
      addBilinear!(U, 3, 4, J34, (0, 1, -1)) 

end

"""
Add interactions to UnitCell object in the local spin frame. 
"""
function addInteractionsLocal!(U::UnitCell, dict::Dict{String, Float64})
        allowed_keys = ["Jxx", "Jyy", "Jzz"]
        for key in allowed_keys
              if !(key in keys(dict))
                    dict[key] = 0.0
              end
        end

        J = [dict["Jxx"] 0 0
             0 dict["Jyy"] 0
             0 0 dict["Jzz"]]
  
        # A neighbours 
        addBilinear!(U, 1, 2, J, (0, 0, 0)) 
        addBilinear!(U, 1, 3, J, (0, 0, 0)) 
        addBilinear!(U, 1, 4, J, (0, 0, 0)) 
        addBilinear!(U, 2, 3, J, (0, 0, 0)) 
        addBilinear!(U, 2, 4, J, (0, 0, 0)) 
        addBilinear!(U, 3, 4, J, (0, 0, 0)) 
  
        # B neighbours
        addBilinear!(U, 1, 2, J, (1, 0, 0)) 
        addBilinear!(U, 1, 3, J, (0, 1, 0)) 
        addBilinear!(U, 1, 4, J, (0, 0, 1)) 
        addBilinear!(U, 2, 3, J, (-1, 1, 0)) 
        addBilinear!(U, 2, 4, J, (-1, 0, 1)) 
        addBilinear!(U, 3, 4, J, (0, 1, -1)) 
  
  end
  
