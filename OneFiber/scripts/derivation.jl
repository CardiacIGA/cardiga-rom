using Plots, LaTeXStrings

Vlv0  = 44
Vwall = 136.  
ΔV  = LinRange(1e-2,1,100) # ΔV = Vlv/Vwall
ΔV0 = Vlv0/Vwall
α   = 1 #.7 #1.15 -> Matches both equation with this value
β1  = 3*α/1.15
β2  = 3*α
#ϕ   = 2.4

## Pressure - Volume relation
plv_σf(ΔV; βstar=1)  = 1/(3*βstar)*log(1 + 1/ΔV )
plv_σf_lin(ΔV)       = (1 + 3*ΔV)^(-1) # <<-- Standard linearization
σf_plv(Vlv; βstar=1) = 1/plv_σf(Vlv, βstar=βstar)
σf_plv_lin(Vlv)      = 1/plv_σf_lin(Vlv)  


f(γ)  = ( log( 1 + 1 / γ ) )^(-1)  
∂f(γ) = ( log( 1 + 1 / γ ) )^(-2) * ( 1 / ( 1 + 1 / γ ) ) * ( γ^(-2) )

βbar(γ)     = (f(γ)/∂f(γ) - γ)
ϕ(βstar, γ) = 3*βstar*(f(γ) - γ*∂f(γ))
β(βstar, γ) = 3*βstar*∂f(γ)

σf_plv_lin_arb(ΔV; β=3, ϕ=1) = (ϕ + β*ΔV)
σf_plv_lin_corr(ΔV; βstar=1, γ=1) = ϕ(βstar, γ)*(1 + (1/βbar(γ))*ΔV)
#σf_plv_lin_corr(ΔV; βstar=1, γ=1) = (ϕ(βstar, γ) + β(βstar, γ)*ΔV)
plv_σf_lin_corr(ΔV; βstar=1, γ=1) = 1 /σf_plv_lin_corr(ΔV; βstar=βstar, γ=γ)  # Corrected 

βstar_in = 1 #1/∂f(0.55)
γ_in     = 0.345 #0.55
color = Dict( :red => "#CC3311", :blue => "#0077BB", :teal => "#009988")
p1 = plot(ΔV,  σf_plv.(ΔV, βstar=βstar_in),                  linewidth=2.5, color=color[:red],  label="\$\\textrm{Nonlinear\\ rsym.,\\ Arts\\ et\\ al}\$")
p3 = plot!(ΔV, σf_plv_lin_corr.(ΔV, βstar=βstar_in, γ=γ_in), linewidth=2.5, color=color[:blue], label="\$\\textrm{Linear\\ rsym.}\$")
p2 = plot!(ΔV, σf_plv_lin.(ΔV),                              linewidth=2.5, color=color[:teal], label="\$\\textrm{Linear\\ cyl.,\\ Arts\\ et\\ al}\$")
# p4 = scatter!([γ_in], [σf_plv_lin_corr.(γ_in, βstar=βstar_in, γ=γ_in)], label="Lin. point")
plot!([ (0.3,1), (0.06,1),], arrow = arrow(:closed, 0.1), color = :black, label=false)
annotate!([0.5], [1], L"3 \mathrm{ln}\left(1 + \frac{V_{\mathrm{w}}}{V}\right)^{-1}")
plot!([ (0.68, 2.2), (0.61,2.78),], arrow = arrow(:closed, 0.1), color = :black, label=false)
annotate!([0.75], [2.], L"1 + 3 \frac{V}{V_{\mathrm{w}}}")
plot!([ (0.6,4.2), (0.9,4.2),], arrow = arrow(:closed, 0.1), color = :black, label=false)
annotate!([0.5], [4.2], L"1 + 3.5 \frac{V}{V_{\mathrm{w}}}")
xlabel!(L"{V}/{V_{\mathrm{w}}}\ \mathrm{[-]}")  #111 #=V/V_{\mathrm{w}}
ylabel!(L"\tau_{\mathrm{fiber}}/p\ \mathrm{[-]}")
# title!(L"\mathrm{Comparison\ pressure-stress\ ratio}")
display(p1)
savefig(p1,"output/figures/Onefiber_comparison_fiberstress.pdf")

## Strain - volume relation
ϵf(ΔV; βstar=1)      = 1/(3*βstar)*( (1 + ΔV)*log(1+ΔV) - ΔV*log(ΔV) ) - 1/(3*βstar)*( (1 + ΔV0)*log(1+ΔV0) - ΔV0*log(ΔV0) ) 
ϵf_lin(ΔV)           = 1/3*( log(1 + 3*ΔV) - log(1 + 3*ΔV0) )
ϵf_lin_corr(ΔV; βstar=1, γ=1) = βbar(γ) / ϕ(βstar, γ)*( log(βbar(γ) + ΔV) - log(βbar(γ) + ΔV0) ) # Corrected
ϵf_lin_arb(ΔV; β=1, ϕ=1) = 1 / β *( log(ϕ + β*ΔV) - log(ϕ + β*ΔV0) ) # Corrected

p1 = plot(ΔV,  ϵf.(ΔV, βstar=βstar_in),                  linewidth=2.5, color=color[:red],  label="\$\\textrm{Nonlinear\\ rsym.,\\ Arts\\ et\\ al}\$")
p3 = plot!(ΔV, ϵf_lin_corr.(ΔV, βstar=βstar_in, γ=γ_in), linewidth=2.5, color=color[:blue], label="\$\\textrm{Linear\\ rsym.}\$")
p2 = plot!(ΔV, ϵf_lin.(ΔV),                              linewidth=2.5, color=color[:teal], label="\$\\textrm{Linear\\ cyl.,\\ Arts\\ et\\ al}\$")
# p4 = scatter!([γ_in], [ϵf_lin_corr.(γ_in, βstar=βstar_in, γ=γ_in)], label="Lin. point")
# title!(L"\mathrm{Comparison\ sarcomere\ strain}")
xlabel!(L"{V}/{V_{\mathrm{w}}}\ \mathrm{[-]}") ##=V/V_{\mathrm{w}}\ 
ylabel!(L"ϵ_{\mathrm{fiber}}  \ \mathrm{[-]}")
display(p1)
savefig(p1,"output/figures/Onefiber_comparison_fiberstrain.pdf")

# ϵf(ΔV)               = 1/3*( (1 + ΔV)*log(1+ΔV) - ΔV*log(ΔV) ) - 1/3*( (1 + ΔV0)*log(1+ΔV0) - ΔV0*log(ΔV0) ) 
# ϵf_corr(ΔV; β=3)     = 1/β*( (1 + ΔV)*log(1+ΔV) - ΔV*log(ΔV) ) - 1/β*( (1 + ΔV0)*log(1+ΔV0) - ΔV0*log(ΔV0) ) 
# ϵf_lin(ΔV)           = 1/3*( log(1 + 3*ΔV) - log(1 + 3*ΔV0) )
# ϵf_lin_corr(ΔV; β=3, ϕ=1) = 1/(β*ϕ)*( log(1 + β*ΔV) - log(1 + β*ΔV0) ) # Corrected

# # ϵf_exp(ΔV)     = 2/(2*ΔV +1)*(ΔV-ΔV0) 
# # ϵf_exp_lin(ΔV) = 3/(3*ΔV +1)*(ΔV-ΔV0) 
# # ϵf_exp(ΔV)     = (1+ΔV*(ΔV+1))*(1+ΔV0*(ΔV0-1))/( (1+ΔV0*(ΔV0+1))*(1+ΔV*(ΔV-1)) )
# # ϵf_exp_lin(ΔV) = 1 + 3*/(3*ΔV0 + 1)*(ΔV-ΔV0) 


# plv_σf(ΔV)               = 1/3*log(1 + 1/ΔV )
# plv_σf_corr(ΔV; β=3)     = 1/β*log(1 + 1/ΔV )  
# plv_σf_lin(ΔV)           = (1 + 3*ΔV)^(-1)  
# plv_σf_lin_corr(ΔV; β=3, ϕ=1) = (1 + β*ΔV)^(-1) / ϕ # Corrected 

# σf_plv(Vlv)               = 1/plv_σf(Vlv)
# σf_plv_corr(Vlv; β=3)     = 1/plv_σf_corr(Vlv, β=β)
# σf_plv_lin(Vlv)           = 1/plv_σf_lin(Vlv)  
# σf_plv_lin_corr(Vlv; β=3, ϕ=1) = 1/plv_σf_lin_corr(Vlv, β=β, ϕ=ϕ) # Corrected  

# ## Strain vs V-ratio
# # p1 = plot(ΔV,  ϵf.(ΔV), label="Analytic")
# # p2 = plot!(ΔV, ϵf_lin.(ΔV), label="Empiric")
# # p3 = plot!(ΔV, ϵf_lin_corr.(ΔV, β=β2), label="Empiric (corr)")
# # p4 = plot!(ΔV, ϵf_corr.(ΔV, β=β1), label="Analytic (corr)")
# # title!(L"\mathrm{Comparison\ sarcomere\ strain}")
# # xlabel!(L"V_{lv}/V_{wall}\ \mathrm{[-]}")
# # ylabel!(L"\epsilon_f\ \mathrm{[-]}")
# # display(p1)

# # ## Stress vs V-ratio
# # p1 = plot(ΔV,  σf_plv.(ΔV), label="Analytic")
# # p2 = plot!(ΔV, σf_plv_lin.(ΔV), label="Empiric")
# # p3 = plot!(ΔV, σf_plv_lin_corr.(ΔV, β=β2), label="Empiric (corr)")
# # p4 = plot!(ΔV, σf_plv_corr.(ΔV, β=β1), label="Analytic (corr)")
# # title!(L"\mathrm{Comparison\ pressure-stress\ ratio}")
# # xlabel!(L"V_{lv}/V_{wall}\ \mathrm{[-]}")
# # ylabel!(L"P_{lv}/\sigma_f\ \mathrm{[-]}")
# # display(p1)

# ϕcor = 1.
# βcor = 1.

# ## Varying β vs V-ratio
# βs = LinRange(0.9, 1.9, 3)
# p1 = plot()
# plot!(ΔV, σf_plv_lin_corr.(ΔV, β=3*βs[1]*βcor, ϕ=ϕcor), label="Empiric (corr)", color=:black, linestyle=:solid)
# plot!(ΔV, σf_plv_corr.(ΔV, β=3*βs[1]), label="Analytic (corr)", color=:blue, linestyle=:dash )
# # plot!(ΔV, σf_plv_lin_corr.(ΔV, β=3*βs[1]), label="Empiric (corr)", color=:black, linestyle=:solid)
# # plot!(ΔV, σf_plv_corr.(ΔV, β=3*βs[1]/1.15), label="Analytic (corr)", color=:blue, linestyle=:dash )
# for βi in βs[2:end]
#     plot!(ΔV, σf_plv_lin_corr.(ΔV, β=3*βi*βcor, ϕ=ϕcor), label=false, color=:black, linestyle=:solid )#, label="Empiric (corr)")
#     plot!(ΔV, σf_plv_corr.(ΔV, β=3*βi), label=false, color=:blue, linestyle=:dash )#, label="Analytic (corr)")
#     # plot!(ΔV, σf_plv_lin_corr.(ΔV, β=3*βi), label=false, color=:black, linestyle=:solid )#, label="Empiric (corr)")
#     # plot!(ΔV, σf_plv_corr.(ΔV, β=3*βi/1.15), label=false, color=:blue, linestyle=:dash )#, label="Analytic (corr)")
# end
# title!(L"\mathrm{Comparison\ pressure-stress\ ratio}")
# xlabel!(L"V_{lv}/V_{wall}\ \mathrm{[-]}")
# ylabel!(L"P_{lv}/\sigma_f\ \mathrm{[-]}")
# display(p1)


# p1 = plot()
# plot!(ΔV, ϵf_corr.(ΔV, β=3*βs[1]), label="Empiric (corr)", color=:black, linestyle=:solid)
# #plot!(ΔV, ϵf_lin_corr.(ΔV, β=3*βs[1]/1.15), label="Analytic (corr)", color=:blue, linestyle=:dash )
# plot!(ΔV, ϵf_lin_corr.(ΔV, β=3*βs[1]*βcor, ϕ=ϕcor), label="Analytic (corr)", color=:blue, linestyle=:dash )
# for βi in βs[2:end]
#     plot!(ΔV, ϵf_corr.(ΔV, β=3*βi), label=false, color=:black, linestyle=:solid )#, label="Empiric (corr)")
#     plot!(ΔV, ϵf_lin_corr.(ΔV, β=3*βi*βcor, ϕ=ϕcor), label=false, color=:blue, linestyle=:dash )#, label="Analytic (corr)")
#     # plot!(ΔV, ϵf_lin_corr.(ΔV, β=3*βi/1.15), label=false, color=:blue, linestyle=:dash )#, label="Analytic (corr)")
# end
# title!(L"\mathrm{Comparison\ sarcomere\ strain}")
# xlabel!(L"V_{lv}/V_{wall}\ \mathrm{[-]}")
# ylabel!(L"\epsilon_f\ \mathrm{[-]}")
# display(p1)