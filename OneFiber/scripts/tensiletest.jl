using Plots, LaTeXStrings

# strain
λ = LinRange(1,1.4,100)

function constitutive_model(λ; C=0.4*1e3, a1=3, a2=6, a3=3)


    # Green-lagrange strains
    F = [λ sqrt.(λ).^(-1) sqrt.(λ).^(-1)]
    E = 0.5*(F.*F .- 1)
    E11 = E[:,1]    # 0.5*(λ^2 - 1)
    E22 = E33 = E[:,2] # 0.5*(1/λ - 1)

    # Invariants
    I1 = E11 + E22 + E33
    I2 = E11.*E22 + E11.*E33 + E22.*E33
    I4 = E11

    Q       = a1.*I1.^2 - a2.*I2 + a3.*I4.^2
    ψ       = C.*(exp.(Q) .- 1)
    ∂I1_∂E  = [ones(size(λ)) ones(size(λ)) ones(size(λ))]
    ∂I2_∂E  = [E22 + E33 E11 + E33 E22 + E11]
    ∂I4_∂E  = [ones(size(λ)) zeros(size(λ)) zeros(size(λ))]
    ∂Q_∂E   = 2*a1.*I1.*∂I1_∂E - a2.*∂I2_∂E + 2*a3.*I4.*∂I4_∂E


    ∂ψ_∂E   = C.*exp.(Q).*∂Q_∂E
    σ       = F.*∂ψ_∂E.*F
    return σ
end


function σff_c(λ0; C=0.4*1e3, a1=3, a2=6, a3=3)

    # Green-lagrange strains
    F = [λ0 sqrt(λ0)^(-1) sqrt(λ0)^(-1)]
    E = 0.5*(F.*F .- 1)
    E11 = E[1]    # 0.5*(λ^2 - 1)
    E22 = E33 = E[2] # 0.5*(1/λ - 1)

    # Invariants
    I1 = E11 + E22 + E33
    I2 = E11*E22 + E11*E33 + E22*E33
    I4 = E11

    Q       = a1.*I1.^2 - a2.*I2 + a3.*I4.^2
    ψ       = C.*(exp.(Q) .- 1)
    ∂I1_∂E  = [1 1 1]
    ∂I2_∂E  = [E22 + E33 E11 + E33 E22 + E11]
    ∂I4_∂E  = [1 0 0]
    ∂Q_∂E   = 2*a1*I1*∂I1_∂E - a2*∂I2_∂E + 2*a3*I4*∂I4_∂E


    ∂ψ_∂E   = C*exp(Q)*∂Q_∂E
    σ       = F.*∂ψ_∂E.*F
    σff     = σ[1] - σ[2]  # Subtract because of hydrostatic pressure sigmaff = - P + sigma
    return σff
end


function get_constants(; C=0.4*1e3, a1=3, a2=6, a3=3)
    # Ensure that (λ1-1)=0.5(λ2-1)
    λ1 = 1.18
    λ2 = 1 + 2*(λ1-1) #1.30
    σ1_ff = σff_c(λ1; C=C, a1=a1, a2=a2, a3=a3)
    σ2_ff = σff_c(λ2; C=C, a1=a1, a2=a2, a3=a3)
    
    # Fix ls0
    ls0 = 1.9

    # Get cp and Tp0
    R   = σ2_ff * λ1 / ( σ1_ff * λ2 )
    cp_ = 1/(λ1 - 1) * log( R - 1 )
    cp  = cp_/ls0
    Tp0 = σ1_ff / ( λ1 * ( exp(cp_*(λ1-1)) - 1 ) )
    return Tp0, cp, ls0
end




function onefiber_model(λ; C=0.4*1e3, a1=3, a2=6, a3=3)

    Tp0, cp, ls0 = get_constants(; C=C, a1=a1, a2=a2, a3=a3)
    #Tp0 = 0.9e3
    #cp  = 6
    #ls0 = 2

    Tp = Tp0.*(exp.(cp.*ls0.*(λ .- 1))  .- 1 )
    σ  = λ.*Tp
    #σ = Tp0*ls0*cp.*(λ .- 1)
    return σ
end

# Constants
C  = 0.4*1e3 
a1 = 3
a2 = 6
a3 = 3


σff_const = constitutive_model(λ; C=C, a1=a1, a2=a2, a3=a3)
σff_const = σff_const[:,1] - σff_const[:,2]
σff_onef  = onefiber_model(λ; C=C, a1=a1, a2=a2, a3=a3)



## Plotting
plot(λ, σff_const.*1e-3, ylabel=L"\mathrm{Cauchy\ stress}\ \sigma_{ff}\ \mathrm{[kPa]}", xlabel=L"\mathrm{Stretch}\ \lambda\ [-]", label=L"\mathrm{Constitutive\ model}")
plot!(λ, σff_onef.*1e-3, ylabel=L"\mathrm{Cauchy\ stress}\ \sigma_{ff}\ \mathrm{[kPa]}", xlabel=L"\mathrm{Stretch}\ \lambda\ [-]", label=L"\mathrm{One-fiber}")
#plot!(λ, σff_clin.*1e-3, ylabel=L"\mathrm{Cauchy\ stress}\ \sigma_{ff}", xlabel=L"\mathrm{Stretch}\ \lambda\ [-]", label=L"C(\lambda)")
