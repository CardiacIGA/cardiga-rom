using OneFiber

Base.@kwdef struct ReferenceValues
    Vlv_ref::Float64
    plv_ref::Float64
    part_ref::Float64
    ta_ref::Float64
    lc_ref::Float64
end

function validate()
    c = Constants()
    ∂ℛ, ℛ = getResidual_matrix_vector(c)


    # ref values
    ref = ReferenceValues(Vlv_ref=44*1e-3, plv_ref=10, part_ref=10e3, ta_ref=0., lc_ref=1.5)

    ## Check ∂ℛVlv
    # check_∂ℛVlv_∂Vlv(∂ℛ, ℛ, ref) # ✓
    # check_∂ℛVlv_∂lc(∂ℛ, ℛ, ref)  # ✓
    # check_∂ℛVlv_∂plv(∂ℛ, ℛ, ref) # ✓
    
    ## Check ∂ℛlc
    check_∂ℛlc_∂Vlv(∂ℛ, ℛ, ref) # ✓
    # check_∂ℛlc_∂lc(∂ℛ, ℛ, ref)  # ✓
    
    ## Check ∂ℛplv
    # check_∂ℛplv_∂Vlv(∂ℛ, ℛ, ref)  # ✓
    # check_∂ℛplv_∂plv(∂ℛ, ℛ, ref)  # ✓
    # check_∂ℛplv_∂part(∂ℛ, ℛ, ref) # ✓

    ## Check ∂ℛpart
    #check_∂ℛpart_∂Vlv(∂ℛ, ℛ, ref)  # ✓
    #check_∂ℛpart_∂plv(∂ℛ, ℛ, ref)  # ✓
    #check_∂ℛpart_∂part(∂ℛ, ℛ, ref) # ✓
end


## --------------------------------------- ##
##               Residual Vlv              ##
## --------------------------------------- ##
function check_∂ℛVlv_∂Vlv(∂ℛ, ℛ, ref)

    (; Vlv_ref, plv_ref, part_ref, ta_ref, lc_ref) = ref

    lc_ref    = 1.6
    ta_ref    = 0.1
    dVlv      = 0.01
    Vlv_range = (44:dVlv:50).*1e-3

    dℛdVlv     = zeros(Float64, length(Vlv_range)-1)
    ∂ℛVlv_∂Vlv = zeros(Float64, length(Vlv_range)-1)
    X0_Vlv      = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        
    for (i, (Vlv_d1, Vlv_d2)) in enumerate(zip(Vlv_range[1:end-1], Vlv_range[2:end]))
        X_Vlv_d1  = Quantities(Vlv=Vlv_d1, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        X_Vlv_d2  = Quantities(Vlv=Vlv_d2, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        dℛdVlv[i] = ( ℛ(X_Vlv_d2, X0_Vlv)[1] - ℛ(X_Vlv_d1, X0_Vlv)[1] )/ (Vlv_d2-Vlv_d1)
        ∂ℛVlv_∂Vlv[i] = ∂ℛ(X_Vlv_d1, X0_Vlv)[1,1]
    end

    println("Norm difference: $(norm(dℛdVlv - ∂ℛVlv_∂Vlv))")
    plot(Vlv_range[1:end-1], [dℛdVlv, ∂ℛVlv_∂Vlv], label=["dℛdVlv" "∂ℛVlv_∂Vlv"], linewidth=3)
end

function check_∂ℛVlv_∂lc(∂ℛ, ℛ, ref)

    (; Vlv_ref, plv_ref, part_ref, ta_ref, lc_ref) = ref

    ta_ref   = 0.1
    dlc      = 0.00001
    lc_range = (1.3:dlc:2.4)

    dℛdlc     = zeros(Float64, length(lc_range)-1)
    ∂ℛVlv_∂lc = zeros(Float64, length(lc_range)-1)
    X0_lc      = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        
    for (i, (lc_d1, lc_d2)) in enumerate(zip(lc_range[1:end-1], lc_range[2:end]))
        X_lc_d1  = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_d1)
        X_lc_d2  = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_d2)
        dℛdlc[i] = ( ℛ(X_lc_d2, X0_lc)[1] - ℛ(X_lc_d1, X0_lc)[1] )/ (lc_d2-lc_d1)
        ∂ℛVlv_∂lc[i] = ∂ℛ(X_lc_d1, X0_lc)[1,2]
    end

    println("Norm difference: $(norm(dℛdlc - ∂ℛVlv_∂lc))")
    plot(lc_range[1:end-1], [dℛdlc, ∂ℛVlv_∂lc], label=["dℛdlc" "∂ℛVlv_∂lc"], linewidth=3)
end


function check_∂ℛVlv_∂plv(∂ℛ, ℛ, ref)

    (; Vlv_ref, plv_ref, part_ref, ta_ref, lc_ref) = ref

    dplv      = 10
    plv_range = (0:dplv:100)

    dℛdplv     = zeros(Float64, length(plv_range)-1)
    ∂ℛVlv_∂plv = zeros(Float64, length(plv_range)-1)
    X0_plv      = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        
    for (i, (plv_d1, plv_d2)) in enumerate(zip(plv_range[1:end-1], plv_range[2:end]))
        X_plv_d1  = Quantities(Vlv=Vlv_ref, plv=plv_d1, part=part_ref, ta=ta_ref, lc=lc_ref)
        X_plv_d2  = Quantities(Vlv=Vlv_ref, plv=plv_d2, part=part_ref, ta=ta_ref, lc=lc_ref)
        dℛdplv[i] = ( ℛ(X_plv_d2, X0_plv)[1] - ℛ(X_plv_d1, X0_plv)[1] )/ (plv_d2-plv_d1)
        ∂ℛVlv_∂plv[i] = ∂ℛ(X_plv_d1, X0_plv)[1,3]
    end

    println("Norm difference: $(norm(dℛdplv - ∂ℛVlv_∂plv))")
    plot(plv_range[1:end-1], [dℛdplv, ∂ℛVlv_∂plv], label=["dℛdplv" "∂ℛVlv_∂plv"], linewidth=3)
end



## --------------------------------------- ##
##               Residual lc               ##
## --------------------------------------- ##
function check_∂ℛlc_∂Vlv(∂ℛ, ℛ, ref)

    (; Vlv_ref, plv_ref, part_ref, ta_ref, lc_ref) = ref

    ta_ref    = 0.1
    dVlv      = 0.001
    Vlv_range = (44:dVlv:50).*1e-3

    dℛdVlv     = zeros(Float64, length(Vlv_range)-1)
    ∂ℛlc_∂Vlv = zeros(Float64, length(Vlv_range)-1)
    X0_Vlv      = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        
    for (i, (Vlv_d1, Vlv_d2)) in enumerate(zip(Vlv_range[1:end-1], Vlv_range[2:end]))
        X_Vlv_d1  = Quantities(Vlv=Vlv_d1, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        X_Vlv_d2  = Quantities(Vlv=Vlv_d2, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        dℛdVlv[i] = ( ℛ(X_Vlv_d2, X0_Vlv)[2] - ℛ(X_Vlv_d1, X0_Vlv)[2] )/ (Vlv_d2-Vlv_d1)
        ∂ℛlc_∂Vlv[i] = ∂ℛ(X_Vlv_d1, X0_Vlv)[2,1]
    end

    println("Norm difference: $(norm(dℛdVlv - ∂ℛlc_∂Vlv))")
    plot(Vlv_range[1:end-1], [dℛdVlv, ∂ℛlc_∂Vlv], label=["dℛdVlv" "∂ℛlc_∂Vlv"], linewidth=3)
end

function check_∂ℛlc_∂lc(∂ℛ, ℛ, ref)

    (; Vlv_ref, plv_ref, part_ref, ta_ref, lc_ref) = ref

    ta_ref   = 0.1
    dlc      = 0.001
    lc_range = (1.3:dlc:2.4)

    dℛdlc     = zeros(Float64, length(lc_range)-1)
    ∂ℛlc_∂lc  = zeros(Float64, length(lc_range)-1)
    X0_lc      = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        
    for (i, (lc_d1, lc_d2)) in enumerate(zip(lc_range[1:end-1], lc_range[2:end]))
        X_lc_d1   = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_d1)
        X_lc_d2   = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_d2)
        dℛdlc[i] = ( ℛ(X_lc_d2, X0_lc)[2] - ℛ(X_lc_d1, X0_lc)[2] )/ (lc_d2-lc_d1)
        ∂ℛlc_∂lc[i] = ∂ℛ(X_lc_d1, X0_lc)[2,2]
    end

    println("Norm difference: $(norm(dℛdlc - ∂ℛlc_∂lc))")
    plot(lc_range[1:end-1], [dℛdlc, ∂ℛlc_∂lc], label=["dℛdlc" "∂ℛlc_∂lc"], linewidth=3)
end




## --------------------------------------- ##
##               Residual plv              ##
## --------------------------------------- ##
function check_∂ℛplv_∂Vlv(∂ℛ, ℛ, ref)

    (; Vlv_ref, plv_ref, part_ref, ta_ref, lc_ref) = ref

    ta_ref    = 0.1
    dVlv      = 0.001
    Vlv_range = (44:dVlv:50).*1e-3

    dℛdVlv     = zeros(Float64, length(Vlv_range)-1)
    ∂ℛplv_∂Vlv = zeros(Float64, length(Vlv_range)-1)
    X0_Vlv      = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        
    for (i, (Vlv_d1, Vlv_d2)) in enumerate(zip(Vlv_range[1:end-1], Vlv_range[2:end]))
        X_Vlv_d1  = Quantities(Vlv=Vlv_d1, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        X_Vlv_d2  = Quantities(Vlv=Vlv_d2, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        dℛdVlv[i] = ( ℛ(X_Vlv_d2, X0_Vlv)[3] - ℛ(X_Vlv_d1, X0_Vlv)[3] )/ (Vlv_d2-Vlv_d1)
        ∂ℛplv_∂Vlv[i] = ∂ℛ(X_Vlv_d1, X0_Vlv)[3,1]
    end

    println("Norm difference: $(norm(dℛdVlv - ∂ℛplv_∂Vlv))")
    plot(Vlv_range[1:end-1], [dℛdVlv, ∂ℛplv_∂Vlv], label=["dℛdVlv" "∂ℛplv_∂Vlv"], linewidth=3)
end

function check_∂ℛplv_∂plv(∂ℛ, ℛ, ref)

    (; Vlv_ref, plv_ref, part_ref, ta_ref, lc_ref) = ref

    dplv      = 0.1
    plv_range = (0:dplv:100)

    dℛdplv     = zeros(Float64, length(plv_range)-1)
    ∂ℛplv_∂plv = zeros(Float64, length(plv_range)-1)
    X0_plv      = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        
    for (i, (plv_d1, plv_d2)) in enumerate(zip(plv_range[1:end-1], plv_range[2:end]))
        X_plv_d1  = Quantities(Vlv=Vlv_ref, plv=plv_d1, part=part_ref, ta=ta_ref, lc=lc_ref)
        X_plv_d2  = Quantities(Vlv=Vlv_ref, plv=plv_d2, part=part_ref, ta=ta_ref, lc=lc_ref)
        dℛdplv[i] = ( ℛ(X_plv_d2, X0_plv)[3] - ℛ(X_plv_d1, X0_plv)[3] )/ (plv_d2-plv_d1)
        ∂ℛplv_∂plv[i] = ∂ℛ(X_plv_d1, X0_plv)[3,3]
    end

    println("Norm difference: $(norm(dℛdplv - ∂ℛplv_∂plv))")
    plot(plv_range[1:end-1], [dℛdplv, ∂ℛplv_∂plv], label=["dℛdplv" "∂ℛplv_∂plv"], linewidth=3)
end

function check_∂ℛplv_∂part(∂ℛ, ℛ, ref)

    (; Vlv_ref, plv_ref, part_ref, ta_ref, lc_ref) = ref

    ta_ref = 0.1
    dpart      = 10
    part_range = (20:dpart:10000)

    dℛdplv     = zeros(Float64, length(part_range)-1)
    ∂ℛplv_∂part = zeros(Float64, length(part_range)-1)
    X0_part      = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        
    for (i, (part_d1, part_d2)) in enumerate(zip(part_range[1:end-1], part_range[2:end]))
        X_part_d1  = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_d1, ta=ta_ref, lc=lc_ref)
        X_part_d2  = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_d2, ta=ta_ref, lc=lc_ref)
        dℛdplv[i] = ( ℛ(X_part_d2, X0_part)[3] - ℛ(X_part_d1, X0_part)[3] )/ (part_d2-part_d1)
        ∂ℛplv_∂part[i] = ∂ℛ(X_part_d1, X0_part)[3,4]
    end

    println("Norm difference: $(norm(dℛdplv - ∂ℛplv_∂part))")
    plot(part_range[1:end-1], [dℛdplv, ∂ℛplv_∂part], label=["dℛdpart" "∂ℛplv_∂part"], linewidth=3)
end



## --------------------------------------- ##
##               Residual part             ##
## --------------------------------------- ##
function check_∂ℛpart_∂Vlv(∂ℛ, ℛ, ref)

    (; Vlv_ref, plv_ref, part_ref, ta_ref, lc_ref) = ref

    ta_ref    = 0.1
    dVlv      = 0.001
    Vlv_range = (44:dVlv:50).*1e-3

    dℛdVlv     = zeros(Float64, length(Vlv_range)-1)
    ∂ℛart_∂Vlv = zeros(Float64, length(Vlv_range)-1)
    X0_Vlv      = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        
    for (i, (Vlv_d1, Vlv_d2)) in enumerate(zip(Vlv_range[1:end-1], Vlv_range[2:end]))
        X_Vlv_d1  = Quantities(Vlv=Vlv_d1, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        X_Vlv_d2  = Quantities(Vlv=Vlv_d2, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        dℛdVlv[i] = ( ℛ(X_Vlv_d2, X0_Vlv)[4] - ℛ(X_Vlv_d1, X0_Vlv)[4] )/ (Vlv_d2-Vlv_d1)
        ∂ℛart_∂Vlv[i] = ∂ℛ(X_Vlv_d1, X0_Vlv)[4,1]
    end

    println("Norm difference: $(norm(dℛdVlv - ∂ℛart_∂Vlv))")
    plot(Vlv_range[1:end-1], [dℛdVlv, ∂ℛart_∂Vlv], label=["dℛdVlv" "∂ℛpart_∂Vlv"], linewidth=3)
end

function check_∂ℛpart_∂plv(∂ℛ, ℛ, ref)

    (; Vlv_ref, plv_ref, part_ref, ta_ref, lc_ref) = ref

    part_ref = -1
    dplv      = 0.1
    plv_range = (0:dplv:100)

    dℛdplv      = zeros(Float64, length(plv_range)-1)
    ∂ℛpart_∂plv = zeros(Float64, length(plv_range)-1)
    X0_plv       = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        
    for (i, (plv_d1, plv_d2)) in enumerate(zip(plv_range[1:end-1], plv_range[2:end]))
        X_plv_d1  = Quantities(Vlv=Vlv_ref, plv=plv_d1, part=part_ref, ta=ta_ref, lc=lc_ref)
        X_plv_d2  = Quantities(Vlv=Vlv_ref, plv=plv_d2, part=part_ref, ta=ta_ref, lc=lc_ref)
        dℛdplv[i] = ( ℛ(X_plv_d2, X0_plv)[4] - ℛ(X_plv_d1, X0_plv)[4] )/ (plv_d2-plv_d1)
        ∂ℛpart_∂plv[i] = ∂ℛ(X_plv_d1, X0_plv)[4,3]
    end

    println("Norm difference: $(norm(dℛdplv - ∂ℛpart_∂plv))")
    plot(plv_range[1:end-1], [dℛdplv, ∂ℛpart_∂plv], label=["dℛdplv" "∂ℛpart_∂plv"], linewidth=3)
end

function check_∂ℛpart_∂part(∂ℛ, ℛ, ref)

    (; Vlv_ref, plv_ref, part_ref, ta_ref, lc_ref) = ref

    ta_ref = 0.1
    dpart      = 10
    part_range = (20:dpart:10000)

    dℛdplv       = zeros(Float64, length(part_range)-1)
    ∂ℛpart_∂part = zeros(Float64, length(part_range)-1)
    X0_part       = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_ref, ta=ta_ref, lc=lc_ref)
        
    for (i, (part_d1, part_d2)) in enumerate(zip(part_range[1:end-1], part_range[2:end]))
        X_part_d1  = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_d1, ta=ta_ref, lc=lc_ref)
        X_part_d2  = Quantities(Vlv=Vlv_ref, plv=plv_ref, part=part_d2, ta=ta_ref, lc=lc_ref)
        dℛdplv[i] = ( ℛ(X_part_d2, X0_part)[4] - ℛ(X_part_d1, X0_part)[4] )/ (part_d2-part_d1)
        ∂ℛpart_∂part[i] = ∂ℛ(X_part_d1, X0_part)[4,4]
    end

    println("Norm difference: $(norm(dℛdplv - ∂ℛpart_∂part))")
    plot(part_range[1:end-1], [dℛdplv, ∂ℛpart_∂part], label=["dℛdpart" "∂ℛpart_∂part"], linewidth=3)
end

validate()