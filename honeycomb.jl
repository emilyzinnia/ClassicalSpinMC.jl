using ClassicalSpinMC

const high_symmetry_points = Dict("K"=>[4pi/3, 0], 
                                   "K'"=>[2pi/3, 2pi/sqrt(3)],
                                   "G0"=>[0.0,0.0],
                                   "G1"=>[2pi,2pi/sqrt(3)],
                                   "M"=>[pi, pi/sqrt(3)],
                                   "M'"=>[pi, -pi/sqrt(3)])

function addInteractionsCartesian!(H::UnitCell, dict::Dict{String, Float64})
      allowed_keys = ["J1xy", "J1xy", "J1z", "D", "E", "J3xy","J3xy","J3z"]
      for key in allowed_keys
            if !(key in keys(dict))
                  dict[key] = 0.0
            end
      end
      U = [-1/2 -sqrt(3)/2 0.0
         sqrt(3)/2 -1/2 0.0
         0.0 0.0 1.0]
      J1_z = [dict["J1xy"]+dict["D"] dict["E"] 0.0 
            dict["E"] dict["J1xy"]+dict["D"] 0.0
            0.0 0.0 dict["J1z"] ]
      J1_y = transpose(U) * J1_z * U
      J1_x = U * J1_z * transpose(U)
      J3 = [dict["J3xy"] 0.0 0.0 
            0.0 dict["J3xy"] 0.0
            0.0 0.0 dict["J3z"]]

      # nearest neighbour interactions 
      addInteraction!(H, 1, 2, J1_x, (0, -1)) #x-bond 
      addInteraction!(H, 1, 2, J1_y, (1, -1)) #y-bond 
      addInteraction!(H, 1, 2, J1_z, (0, 0 )) #z-bond 

      # third nearest neighbour interactions 
      addInteraction!(H, 1, 2, J3, (1, 0)) #x-bond 
      addInteraction!(H, 1, 2, J3, (1, -2)) #y-bond 
      addInteraction!(H, 1, 2, J3, (-1, 0)) #z-bond 
      end

function addInteractionsKitaev!(H::UnitCell, dict::Dict{String, Float64})
      allowed_keys = ["J", "K", "G", "Gp", "J3"]
      for key in allowed_keys
            if !(key in keys(dict))
                  dict[key] = 0.0
            end
      end
      J = dict["J"]
      K = dict["K"]
      G = dict["G"]    
      Gp = dict["Gp"]
      J3 = dict["J3"]
      # xbond
      Jx = [K+J Gp Gp
            Gp J G
            Gp G J]
      # ybond
      Jy = [J Gp G
            Gp K+J Gp 
            G Gp J]
      # zbond
      Jz = [J G Gp 
            G J Gp
            Gp Gp K+J]

      J3 = [J3 0.0 0.0 
            0.0 J3 0.0
            0.0 0.0 J3]

      # nearest neighbour interactions 
      addInteraction!(H, 1, 2, Jx, (0, -1)) #x-bond 
      addInteraction!(H, 1, 2, Jy, (1, -1)) #y-bond 
      addInteraction!(H, 1, 2, Jz, (0, 0 )) #z-bond 

      # third nearest neighbour interactions 
      addInteraction!(H, 1, 2, J3, (1, 0)) #x-bond 
      addInteraction!(H, 1, 2, J3, (1, -2)) #y-bond 
      addInteraction!(H, 1, 2, J3, (-1, 0)) #z-bond 
end




