
Base.@kwdef mutable struct Constants
    # Unit conversion
    kPa::Float64  = 1e3
    ml::Float64   = 1e-3
    ms::Float64   = 1e-3
    s::Float64    = 1
    mmHg::Float64 = 133.322368

    # Scaling factors
    # α::Float64 = 1.
    # β::Float64 = 1.
    # γ::Float64 = 1.
    # λ::Float64 = 1.
    # ϕ::Float64 = 1.

    α::Float64 = 1.
    β::Float64 = 1.
    γ::Float64 = 1.
    λ::Float64 = 1.
    ϕ::Float64 = 1.
    ω::Float64 = 1.

    # Passive Constants
    Tp0::Float64    = 0.4*kPa # [Pa]
    cp::Float64     = 6.0     # [-]
    ls0::Float64    = 1.9     # [μm]
    Vlv0::Float64   = 44*ml   # [l]
    Vwall::Float64  = 136*ml  # [l]

    # Active Constants
    Ta0::Float64    = 140.0*kPa # [Pa]
    Ea::Float64     =  20.0     # [1/μm]
    lc0::Float64    =   1.5     # [μm]
    ld::Float64     =  -1.0     # [μm]
    τr::Float64     =  75.0*ms  # [s]
    τd::Float64     = 150.0*ms  # [s]
    al::Float64	    =   2.0     # [1/μm]
    b::Float64	    = 160.0*ms  # [s/μm]
    v0::Float64	    =   7.5     # [μm/s]
    tstart::Float64 = 300*ms    # [s]
    tcycle::Float64 = 800*ms    # [s]
    trelax::Float64 = 298*ms    # [s]

    # Circulatory Constants
    Rart::Float64   =    0.010*kPa*s/ml # [Pa⋅s/l]
    Rper::Float64   =    0.120*kPa*s/ml # [Pa⋅s/l]
    Rven::Float64   =    0.002*kPa*s/ml # [Pa⋅s/l]
    Cart::Float64   =   25.0  *ml/kPa   # [l/Pa]
    Cven::Float64   =  600.0  *ml/kPa   # [l/Pa]
    Vart0::Float64  =  500.0  *ml       # [l]
    Vven0::Float64  = 3000.0  *ml       # [l]
    Vtotal::Float64 = 5000.0  *ml       # [l]
    
    # Time integration Constants
    δt::Float64 = 2.0*ms # [s]
    θ::Float64  = 0.5    # [-]

end

# struct Parameters
#     α
#     β
#     γ
#     λ
#     ϕ
# end

# mutable struct ConstantsV2
#     # Unit conversion
#     kPa::Float64
#     ml::Float64
#     ms::Float64
#     s::Float64
#     mmHg::Float64

#     # Passive Constants
#     Tp0::Float64
#     cp::Float64
#     ls0::Float64
#     Vlv0::Float64
#     Vwall::Float64

#     # Active Constants
#     Ta0::Float64
#     Ea::Float64
#     lc0::Float64
#     ld::Float64
#     τr::Float64
#     τd::Float64
#     al::Float64
#     b::Float64
#     v0::Float64
#     tstart::Float64
#     tcycle::Float64
#     trelax::Float64

#     # Circulatory Constants
#     Rart::Float64
#     Rper::Float64
#     Rven::Float64
#     Cart::Float64
#     Cven::Float64 
#     Vart0::Float64
#     Vven0::Float64
#     Vtotal::Float64
    
#     # Time integration Constants
#     δt::Float64
#     θ::Float64

#     α
#     β
#     γ
#     λ
#     ϕ

#     function ConstantsV2(params::Parameters, constants::Constants)

#         ## Constants---------------------------------------------------------------------
#         (; Vwall, ls0, lc0, Vlv0, cp, Tp0, al, Ta0, b, ld, τr, τd, Ea, v0,
#         tstart, tcycle, trelax, 
#         Cven, Rper, Rart, Rven, Vven0, Vtotal, Vart0, Cart,
#         δt, θ,
#         kPa, ml, ms, s, mmHg) = constants

#         new(kPa,
#             ml,
#             ms,
#             s,
#             mmHg,

#             Tp0,    # [Pa]
#             cp,     # [-]
#             ls0,    # [μm]
#             Vlv0,   # [l]
#             Vwall,  # [l]

#             Ta0,    # [Pa]
#             Ea,    # [1/μm]
#             lc0,    # [μm]
#             ld,    # [μm]
#             τr,    # [s]
#             τd,    # [s]
#             al,    # [1/μm]
#             b,    # [s/μm]
#             v0,    # [μm/s]
#             tstart,  # [s]
#             tcycle,  # [s]
#             trelax,  # [s]

#             Rart, # [Pa⋅s/l]
#             Rper, # [Pa⋅s/l]
#             Rven, # [Pa⋅s/l]
#             Cart, # [l/Pa]
#             Cven, # [l/Pa]
#             Vart0, # [l]
#             Vven0, # [l]
#             Vtotal, # [l]
            
#             # Time integration Constants
#             δt,
#             θ,

#             params.α,
#             params.β,
#             params.γ,
#             params.λ,
#             params.ϕ)
#     end
# end

# Keyword based struct
Base.@kwdef mutable struct Quantities
    ta::Float64
    Vlv::Float64
    lc::Float64
    plv::Float64
    part::Float64
end
# Base.copy(s::S) = S(s.x, s.y)
# Base.@kwdef mutable struct Quantities
#     ta::Real
#     Vlv::Real
#     lc::Real
#     plv::Real
#     part::Real
# end

Base.@kwdef struct Functions
    ls::Function
    Tpas::Function
    Tact::Function
    fiso::Function
    ftwitch::Function
    tmax::Function
    Vart::Function
    pven::Function
    qart::Function
    qven::Function
    qper::Function
end



function getResidual_matrix_vector(constants; return_eval=false)

    ## Constants---------------------------------------------------------------------
    (; Vwall, ls0, lc0, Vlv0, cp, Tp0, al, Ta0, b, ld, τr, τd, Ea, 
        Cven, Rper, Rart, Rven, Vven0, Vtotal, Vart0, Cart,
        δt, θ, v0,
        α, β, γ, λ, ϕ,# Scaling factors
        kPa, ml, ms, s) = constants

    # Rescale specific quantities
    #ls0 = α*ls0   # Sarcomere length
    ϕg   = 2*ϕ
    βg   = 3*β     # Geometric scaling
    T_p0 = γ*Tp0   # Passive stiffness
    c_p  = λ*cp    # Stiffness parameter

    # Translation functionals
    # f(ω)  = ( log( 1 + 1 / ω ) )^(-1)  
    # ∂f(ω) = ( log( 1 + 1 / ω ) )^(-2) * ( 1 / ( 1 + 1 / ω ) ) * ( ω^(-2) )

    # βbar(ω)     = (f(ω)/∂f(ω) - ω)
    # ϕ_(βstar, ω) = 3*βstar*(f(ω) - ω*∂f(ω))
    # #β_(βstar, ω) = 3*βstar*∂f(ω)


    # βg = 1 / βbar(0.55*ω)
    # ϕ  = ϕ_(β, 0.55*ω)


    ## Functions---------------------------------------------------------------------
    #ls(Vlv)   = ls0*( ( 1 + 3*( Vlv / Vwall ) ) / ( 1 + 3*( Vlv0 / Vwall ) ) )^( 1 // 3 )
    
    # Option 1:
    #ls(Vlv)   = α*ls0*( ( 1 + βg*( Vlv / Vwall ) ) / ( 1 + βg*( Vlv0 / Vwall ) ) )^( 1 / βg )
    # Option 2:
    #ls(Vlv)   = ls0*( ( 1 + βg*( Vlv / Vwall ) ) / ( 1 + βg*( Vlv0 / Vwall ) ) )^( α / βg )
    # Option 3:
    #ls(Vlv)   = ls0 * ( ( 1 + βg*( Vlv / Vwall ) ) / ( 1 + βg*( Vlv0 / Vwall ) ) )^( α / ( βg * ϕ ) )
    # Option 3:
    ls(Vlv)   = ls0 * ( ( ϕg + βg*( Vlv / Vwall ) ) / ( ϕg + βg*( Vlv0 / Vwall ) ) )^( 1 / ( βg ) )



    # Passive
    Tpas(Vlv) = T_p0*( exp( c_p*( ls(Vlv) - ls0 ) ) - 1 )*max(sign(ls(Vlv)-ls0), 0)

    # Active
    tmax(Vlv)         = b*( ls(Vlv) - ld )
    fiso(lc)          = Ta0*( tanh( al*( lc - lc0 ) )^2 )*max(sign(lc-lc0), 0)
    ftwitch(Vlv, ta)  = ( tanh( ta / τr )^2 )*( tanh( ( tmax(Vlv) - ta ) / τd )^2 )*max(sign(ta), 0)*max(-sign(ta-tmax(Vlv)), 0)
    Tact(Vlv, lc, ta) = fiso(lc)*ftwitch(Vlv, ta)*Ea*( ls(Vlv) - lc )

    # Sarcomere dynamics
    Flc(Vlv, lc) = ( Ea*( ls(Vlv) - lc ) - 1 )*v0

    # Circulatory 
    Vart(part)            = Cart*part + Vart0                            # Arterial volume
    pven(Vlv, part)       = (Vtotal-Vven0-Vart(part)-Vlv )/Cven          # Venous pressure
    qven(Vlv, plv, part)  = max( (pven(Vlv, part) - plv) /Rven , 0 ) # Venous volume flow
    qart(plv, part)       = max( (plv            - part) /Rart , 0 ) # Arterial volume flow 
    qper(Vlv, part)       = (part - pven(Vlv, part))/Rper # Peripheral volume flow 
    Fplv(Vlv, plv, part)  = qven(Vlv, plv, part)- qart(plv, part)        # Volume flow difference (plv)
    Fpart(Vlv, plv, part) = qart(plv, part)- qper(Vlv, part)             # Volume flow difference (part)

    

    
    ## Derivatives ----------------------------------------------------------------------------------
    # Option 1:
    #∂ls_∂Vlv(Vlv)          = α .* ls0 .* (Vlv .* βg + Vwall) .^ ((1 - βg) ./ βg) .* (Vlv0 .* βg + Vwall) .^ (-1 ./ βg)
    # Option 2:
    #∂ls_∂Vlv(Vlv)          = ls0 .* α .* (Vlv .* βg + Vwall) .^ ((α - βg) ./ βg) .* (Vlv0 .* βg + Vwall) .^ (-α ./ βg)
    # Option 3:
    #∂ls_∂Vlv(Vlv)          = ls0 .* α .* (Vlv .* βg + Vwall) .^ (α ./ (βg .* ϕ) - 1) .* (Vlv0 .* βg + Vwall) .^ ( -α ./ (βg .* ϕ)) ./ ϕ
    # Option 4:
    ∂ls_∂Vlv(Vlv)          = ls0 .* (Vlv .* βg + Vwall .* ϕg) .^ ((1 - βg) ./ βg) .* (Vlv0 .* βg + Vwall .* ϕg) .^ (-1 ./ βg)

    ∂Tp_∂Vlv(Vlv)          = T_p0 .* c_p .* ∂ls_∂Vlv(Vlv) .* exp(c_p .* ( ls(Vlv) - ls0 ))
    #∂Tp_∂Vlv(Vlv)          = T_p0 .* c_p .* ls0 .* (Vlv .* βg + Vwall) .^ ((1 - βg) ./ βg) .* (Vlv0 .* βg + Vwall).^ (-1 ./ βg) .* exp(c_p .* ls0 .* (Vlv0 .* βg + Vwall) .^ (-1 ./ βg) .* ((Vlv .* βg + Vwall) .^ (1 ./ βg) - (Vlv0 .* βg + Vwall) .^ (1 ./ βg)))
    ∂fiso_∂lc(lc)          = (2 * Ta0 .* al .* tanh(al .* (lc - lc0)) ./ cosh(al .* (lc - lc0)) .^ 2)*max(sign(lc-lc0), 0)
    # ∂ftwitch_∂Vlv(Vlv, ta) = 2 * b .* ls0 .* (Vlv .* βg + Vwall) .^ (-(βg - 1) ./ βg) .* (Vlv0 .* βg + Vwall) .^(-1 ./ βg) .* (tanh((b .* ld - b .* ls0 .* (Vlv .* βg + Vwall) .^ (1 ./ βg) 
    #                           .* (Vlv0 .* βg + Vwall) .^ (-1 ./ βg) + ta) ./ τd) .^ 2 - 1) .* tanh(ta ./ τr) .^ 2 .*tanh((b .* ld - b .* ls0 .* (Vlv .* βg + Vwall) .^ (1 ./ βg) .* (Vlv0 .* βg + Vwall) .^ (-1 ./ βg) + ta) ./ τd) ./ τd *max(sign(ta), 0)*max(sign(ta-tmax(Vlv)), 0)
    ∂ftwitch_∂Vlv(Vlv, ta) = ( 2 * b .* tanh(ta ./ τr) .^ 2  ./ τd  ) .* tanh((b .* ( ls(Vlv) - ld ) - ta) ./ τd) .* (1 - tanh((b .* ( ls(Vlv) - ld ) - ta) ./ τd).^2 ).* ∂ls_∂Vlv(Vlv) *max(sign(ta), 0)*max(sign(ta-tmax(Vlv)), 0)


    # ∂ls_∂Vlv(Vlv)          = ls0 ./ ((3 * Vlv + Vwall) .^ (2 // 3) .* (3 * Vlv0 + Vwall) .^ (1 // 3))
    # ∂Tp_∂Vlv(Vlv)          = T_p0 .* c_p .* ls0 .* exp(c_p .* ls0 .* ((3 * Vlv + Vwall) .^ (1 // 3) ./ (3 * Vlv0+ Vwall) .^ (1 // 3) - 1)) ./ ((3 * Vlv + Vwall) .^ (2 // 3) .* (3 * Vlv0 + Vwall) .^ (1 // 3))
    # ∂fiso_∂lc(lc)          = (2 * Ta0 .* al .* tanh(al .* (lc - lc0)) ./ cosh(al .* (lc - lc0)) .^ 2)*max(sign(lc-lc0), 0)
    # ∂ftwitch_∂Vlv(Vlv, ta) = ( 2 * b .* ls0 .* (tanh((b .* ld - b .* ls0 .* (3 * Vlv + Vwall) .^ (1 // 3) ./ (3* Vlv0 + Vwall) .^ (1 // 3) + ta) ./ τd) .^ 2 - 1) 
    #                            .* tanh(ta ./ τr) .^ 2 .* tanh((b .* ld - b .* ls0 .* (3 * Vlv + Vwall) .^ (1 // 3) ./ (3 * Vlv0 + Vwall) .^ (1 // 3) + ta) ./ τd) 
    #                            ./ (τd .* (3 * Vlv + Vwall) .^ (2 // 3) .* (3 * Vlv0 + Vwall) .^ (1 // 3)) )*max(sign(ta), 0)*max(sign(ta-tmax(Vlv)), 0)

    ∂Ta_∂Vlv(Vlv, lc, ta) = fiso(lc)*∂ftwitch_∂Vlv(Vlv, ta)*Ea*( ls(Vlv) - lc ) + fiso(lc)*ftwitch(Vlv, ta)*Ea*∂ls_∂Vlv(Vlv)
    ∂Ta_∂lc(Vlv, lc, ta)  = ∂fiso_∂lc(lc)*ftwitch(Vlv, ta)*Ea*( ls(Vlv) - lc ) - fiso(lc)*ftwitch(Vlv, ta)*Ea

    ∂qper_∂Vlv                  =   1 / ( Rper * Cven )
    ∂qper_∂part                 =   1 / Rper
    ∂qart_∂plv(plv, part)       =   1 / Rart * max(sign(plv-part), 0) # Latter part ensures 0 when flow is reversed, otherwise 1
    ∂qart_∂part(plv, part)      = - 1 / Rart * max(sign(plv-part), 0) # Latter part ensures 0 when flow is reversed, otherwise 1
    ∂qven_∂Vlv(Vlv, plv, part)  = - 1 / ( Rven * Cven ) * max(sign(pven(Vlv, part)-plv), 0) # Latter part ensures 0 when flow is reversed, otherwise 1
    ∂qven_∂plv(Vlv, plv, part)  = - 1 / Rven            * max(sign(pven(Vlv, part)-plv), 0) # Latter part ensures 0 when flow is reversed, otherwise 1
    ∂qven_∂part(Vlv, plv, part) = - Cart / ( Rven * Cven ) * max(sign(pven(Vlv, part)-plv), 0) # Latter part ensures 0 when flow is reversed, otherwise 1




    ## Residuals ------------------------------------------------------------------------------------
    # Scale array
    scale = [1/kPa 1. 1/ml 1/ml]# Scale individual residuals to ≈[unit]

    # Vector: (we use x = param-struct of current time step, x0 = param-struct of previous time step)
    # Option 1: # <<<<<---- USE THIS!
    #ℛVlv(x, x0)  = ls(x.Vlv) / ls0 *( Tpas(x.Vlv) + Tact(x.Vlv, x.lc, x.ta) ) - x.plv*( 1 + βg*x.Vlv / Vwall ) # Note this is for Option 1!
    # Option 2:
    #ℛVlv(x, x0)  = ls(x.Vlv) / ls0 *( Tpas(x.Vlv) + Tact(x.Vlv, x.lc, x.ta) ) - x.plv*( 1 + βg*x.Vlv / Vwall ) / α # Note this is for Option 1!
    # Option 3:
    #ℛVlv(x, x0)  = ls(x.Vlv) / ls0 *( Tpas(x.Vlv) + Tact(x.Vlv, x.lc, x.ta) ) - x.plv*( 1 + βg*x.Vlv / Vwall ) * ϕ # Note this is for Option 1!
    # Option 4:
    ℛVlv(x, x0)  = ls(x.Vlv) / ls0 *( Tpas(x.Vlv) + Tact(x.Vlv, x.lc, x.ta) ) - x.plv*( ϕg + βg*x.Vlv / Vwall ) # Note this is for Option 1!
    ℛlc(x, x0)   =         x.lc  - x0.lc    - θ*δt*Flc(x.Vlv, x.lc)            - (1-θ)*δt*Flc(x0.Vlv, x0.lc) 
    ℛplv(x, x0)  =         x.Vlv - x0.Vlv   - θ*δt*Fplv(x.Vlv, x.plv, x.part)  - (1-θ)*δt*Fplv(x0.Vlv, x0.plv, x0.part)
    ℛpart(x, x0) = Cart*( x.part - x0.part) - θ*δt*Fpart(x.Vlv, x.plv, x.part) - (1-θ)*δt*Fpart(x0.Vlv, x0.plv, x0.part)
    
    ℛ(x, x0)   = [scale[1]*ℛVlv(x,x0) scale[2]*ℛlc(x,x0) scale[3]*ℛplv(x,x0) scale[4]*ℛpart(x,x0)]'

    # Matrix: Residual derivatives (we use x = param-struct of current time step, x0 = param-struct of previous time step)
    # Option 1: <<<--- USE THIS!!
    #∂ℛVlv_∂Vlv(x, x0)  = ( ∂ls_∂Vlv(x.Vlv) / ls0 * ( Tpas(x.Vlv) + Tact(x.Vlv,x.lc,x.ta) ) - βg*x.plv/Vwall) + ( ls(x.Vlv) / ls0 * ( ∂Tp_∂Vlv(x.Vlv) + ∂Ta_∂Vlv(x.Vlv, x.lc, x.ta) ) )
    # Option 2:
    #∂ℛVlv_∂Vlv(x, x0)  = ( ∂ls_∂Vlv(x.Vlv) / ls0 * ( Tpas(x.Vlv) + Tact(x.Vlv,x.lc,x.ta) ) - βg*x.plv/( Vwall*α)) + ( ls(x.Vlv) / ls0 * ( ∂Tp_∂Vlv(x.Vlv) + ∂Ta_∂Vlv(x.Vlv, x.lc, x.ta) ) )
    #∂ℛVlv_∂Vlv(x, x0)  = ( ∂ls_∂Vlv(x.Vlv) / ls0 * ( Tpas(x.Vlv) + Tact(x.Vlv,x.lc,x.ta) ) - βg*ϕ*x.plv/Vwall) + ( ls(x.Vlv) / ls0 * ( ∂Tp_∂Vlv(x.Vlv) + ∂Ta_∂Vlv(x.Vlv, x.lc, x.ta) ) )
    # Option 4:
    #∂ℛVlv_∂Vlv(x, x0)  = ( ∂ls_∂Vlv(x.Vlv) / ls0 * ( Tpas(x.Vlv) + Tact(x.Vlv,x.lc,x.ta) ) - βg*x.plv/( Vwall*α)) + ( ls(x.Vlv) / ls0 * ( ∂Tp_∂Vlv(x.Vlv) + ∂Ta_∂Vlv(x.Vlv, x.lc, x.ta) ) )
    ∂ℛVlv_∂Vlv(x, x0)  = ( ∂ls_∂Vlv(x.Vlv) / ls0 * ( Tpas(x.Vlv) + Tact(x.Vlv,x.lc,x.ta) ) - βg*x.plv/Vwall) + ( ls(x.Vlv) / ls0 * ( ∂Tp_∂Vlv(x.Vlv) + ∂Ta_∂Vlv(x.Vlv, x.lc, x.ta) ) )
    
    ∂ℛVlv_∂lc(x, x0)   = ls(x.Vlv)/ls0 * ∂Ta_∂lc(x.Vlv, x.lc, x.ta)
    
    #∂ℛVlv_∂plv(x, x0)  = - ( 1 + βg*x.Vlv / Vwall )    # Note this is for Option 1! <<<<<----- USE THIS!
    #∂ℛVlv_∂plv(x, x0)  = - ( 1 + βg*x.Vlv / Vwall )/α  # Option 2!
    # ∂ℛVlv_∂plv(x, x0)  = - ( 1 + βg*x.Vlv / Vwall ) * ϕ # Option 3!
    ∂ℛVlv_∂plv(x, x0)  = - ( ϕg + βg*x.Vlv / Vwall )  # Option 4!
    ∂ℛVlv_∂part(x, x0) = 0.
    
    ∂ℛlc_∂Vlv(x, x0)  = -δt*θ*Ea*v0*∂ls_∂Vlv(x.Vlv)
    ∂ℛlc_∂lc(x, x0)   =  1 + δt*θ*v0*Ea
    ∂ℛlc_∂plv(x, x0)  =  0.
    ∂ℛlc_∂part(x, x0) =  0.

    ∂ℛplv_∂Vlv(x, x0)  =  1-δt*θ*∂qven_∂Vlv(x.Vlv, x.plv, x.part)
    ∂ℛplv_∂lc(x, x0)   =  0.
    ∂ℛplv_∂plv(x, x0)  = -δt*θ*( ∂qven_∂plv(x.Vlv, x.plv, x.part) - ∂qart_∂plv(x.plv, x.part) )
    ∂ℛplv_∂part(x, x0) = -δt*θ*( ∂qven_∂part(x.Vlv, x.plv, x.part)      - ∂qart_∂part(x.plv, x.part) )

    ∂ℛpart_∂Vlv(x, x0)  =  δt*θ*∂qper_∂Vlv
    ∂ℛpart_∂lc(x, x0)   =  0.
    ∂ℛpart_∂plv(x, x0)  = -δt*θ*∂qart_∂plv(x.plv, x.part)
    ∂ℛpart_∂part(x, x0) =  Cart-δt*θ*( ∂qart_∂part(x.plv, x.part) - ∂qper_∂part ) 

    # Stiffness matrix (Newton-Raphson)
    # Scale the matrix accordingly (to ≈[unit])
    ∂ℛ(x, x0) = [scale[1]*∂ℛVlv_∂Vlv(x, x0)  scale[1]*∂ℛVlv_∂lc(x, x0)  scale[1]*∂ℛVlv_∂plv(x, x0)  scale[1]*∂ℛVlv_∂part(x, x0);
                  scale[2]*∂ℛlc_∂Vlv(x, x0)   scale[2]*∂ℛlc_∂lc(x, x0)   scale[2]*∂ℛlc_∂plv(x, x0)   scale[2]*∂ℛlc_∂part(x, x0);
                  scale[3]*∂ℛplv_∂Vlv(x, x0)  scale[3]*∂ℛplv_∂lc(x, x0)  scale[3]*∂ℛplv_∂plv(x, x0)  scale[3]*∂ℛplv_∂part(x, x0);
                  scale[4]*∂ℛpart_∂Vlv(x, x0) scale[4]*∂ℛpart_∂lc(x, x0) scale[4]*∂ℛpart_∂plv(x, x0) scale[4]*∂ℛpart_∂part(x, x0)] 
    if return_eval
        return ∂ℛ, ℛ, Functions(ls=ls, Tpas=Tpas, Tact=Tact, fiso=fiso, ftwitch=ftwitch, tmax=tmax, Vart=Vart, 
                                        pven=pven, qart=qart, qven=qven, qper=qper)
    else
        return ∂ℛ, ℛ
    end

end


function return_valve_points_new(P,V; cycle=1, return_idx=false)# Reutnr (p_lv, V_lv) in following order of instance: [Open Mitral, Close Mitral, Open aorta, Closing Aorta]
    arb_length  = floor(Int, length(P)/400)*6
    start_fill  = Matrix{Float64}(undef, (arb_length,3))
    start_eject = Matrix{Float64}(undef, (arb_length,3))
    end_fill    = Matrix{Float64}(undef, (arb_length,3))
    end_eject   = Matrix{Float64}(undef, (arb_length,3))


    Vmean   = mean(V) # get a reference
    iso_vol = false
    ϵ = 1e-6
    global kee=1
    global kef=1
    global ksf=1
    global kse=1;

    dVdt = diff(V)

    for (i, dv) in enumerate(dVdt)

        if i != 1 # Skip first value? Sometimes initial value is same as first calc. step
            # println("i, ", i)
            # println("dv, ", dv)
            if ( abs(dv) ≤ ϵ ) && ( !iso_vol ) # We are in isovolumetric stage
                # println("dVdt", dVdt[i-1])
                
                if dVdt[i-1] > ϵ # We were ascending before -> End-filling
                    end_fill[kef,:]  = [i P[i] V[i]]
                    global kef += 1
                else # We were descending before -> End-ejection
                    end_eject[kee,:] = [i P[i] V[i]]
                    global kee += 1
                end
                iso_vol = true

            elseif iso_vol # We are in an isovolumetric phase and need to check if the dvdt gradient changes
                if dv > ϵ # We are currently ascending -> Start-filling
                    start_fill[ksf,:]  = [i P[i] V[i]]
                    global ksf += 1
                    iso_vol = false
                elseif dv < -ϵ # We were descending before -> Start-ejection
                    start_eject[kse,:] = [i P[i] V[i]]
                    global kse += 1
                    iso_vol = false
                end
        
            end 
        end
    end

    if return_idx
        return transpose(reshape([start_fill[cycle,:]; end_fill[cycle,:]; start_eject[cycle,:]; end_eject[cycle,:]], 3,4)) 
    else
        return transpose(reshape([start_fill[cycle,2:end]; end_fill[cycle,2:end]; start_eject[cycle,2:end]; end_eject[cycle,2:end]], 2,4))
    end
    
end



function return_valve_points(P,V; cycle=1, return_idx=false)# Reutnr (p_lv, V_lv) in following order of instance: [Open Mitral, Close Mitral, Open aorta, Closing Aorta]
    arb_length  = floor(Int, length(P)/400)*4
    start_fill  = Matrix{Float64}(undef, (arb_length,3))
    start_eject = Matrix{Float64}(undef, (arb_length,3))
    end_fill    = Matrix{Float64}(undef, (arb_length,3))
    end_eject   = Matrix{Float64}(undef, (arb_length,3))


    Vmean   = mean(V) # get a reference
    iso_vol = false
    ϵ = 1e-10
    global kee=1
    global kef=1
    global ksf=1
    global kse=1;
    for (i, (p, v)) in enumerate(zip(P,V))

        if i != 1 # Skip the first index
            
            if abs(V[i-1] - v) ≤ ϵ # We are in an isovolumetric phase
                if !iso_vol # We werent in iso_vol phase before
                    if v < Vmean # We are at the left = end_ejection/begin_filling
                        end_eject[kee,:] = [i p v]
                        global kee += 1
                    else
                        end_fill[kef,:]  = [i p v]
                        global kef += 1
                    end
                end
                iso_vol = true
            else
                if iso_vol # We were in iso_vol phase before
                    if v < Vmean # We are at the left = end_ejection/begin_filling
                        start_fill[ksf,:] = [i p v]
                        global ksf += 1
                    else
                        start_eject[kse,:] = [i p v]
                        global kse += 1
                    end
                end
                iso_vol = false
            
            end
        
            
        end
    end

    if return_idx
        return transpose(reshape([start_fill[cycle,:]; end_fill[cycle,:]; start_eject[cycle,:]; end_eject[cycle,:]], 3,4)) 
    else
        return transpose(reshape([start_fill[cycle,2:end]; end_fill[cycle,2:end]; start_eject[cycle,2:end]; end_eject[cycle,2:end]], 2,4))
    end
    
end