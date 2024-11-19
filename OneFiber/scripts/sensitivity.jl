using OneFiber, Plots, StatsPlots, LaTeXStrings, AxisArrays
using DelimitedFiles, Distributions, LinearAlgebra

# Initialize input values
constants = Constants()
(; ms, ml, kPa, mmHg, tcycle, δt) = constants

# Set nr of cycles to be computed
ncycles = 1
#idx_range = ((ncycles-1)*Int(tcycle/δt) + 1):(ncycles*Int(tcycle/δt))

# Set range of input constants
#αs = range(0.3, 1.7, 15)
βs = range(0.3, 1.7, 15)
γs = range(0.3, 1.7, 15)
λs = range(0.3, 1.7, 15)
ϕs = range(0.2, 4, 15)

# Loop over α parameter
function parameter_loop(param, param_range)
    sens = Dict("Plv" => Array{Float64}(undef, size(βs,1), Int(tcycle/δt)+1),
                "Vlv" => Array{Float64}(undef, size(βs,1), Int(tcycle/δt)+1))

    for (i, par) in enumerate(param_range)
        setproperty!(constants, param, par)
        time, Vlvmean, lc, plvmean, part = SolveOneFiber(constants, ncycles=ncycles, return_eval=false, print_info=false)

        sens["Plv"][i,:] = plvmean/mmHg 
        sens["Vlv"][i,:] = Vlvmean/ml
    end
    setproperty!(constants, param, 1)
    return sens
end

# Run simulations for all ranges
#sens_α = parameter_loop(:α, αs)
sens_β = parameter_loop(:β, βs)
sens_γ = parameter_loop(:γ, γs)
sens_λ = parameter_loop(:λ, λs)
sens_ϕ = parameter_loop(:ϕ, ϕs)

function plot_sens(sens, param_range; xlabel="", ylabel="", title="", arrow=[[0,0],[120, 120]], ylims=[0,130], xlims=[0, 120], legend=false)
    p = plot(legend=legend)
    for (i, par) in enumerate(param_range)
        plot!(sens["Vlv"][i,:], sens["Plv"][i,:], label=string(par))
    end
    plot!(arrow[1], arrow[2], arrow=true,color=:black, linewidth=2)#,label="")
    ylabel!(ylabel)
    xlabel!(xlabel)
    ylims!(ylims[1], ylims[2])
    xlims!(xlims[1], xlims[2])
    title!(title)
    return p
end

function parameter_loop_2(param1, param2, param_range1, param_range2)
    sens = Dict("Plv" => Array{Float64}(undef, size(βs,1), Int(tcycle/δt)+1),
                "Vlv" => Array{Float64}(undef, size(βs,1), Int(tcycle/δt)+1))

    for (i, (par1, par2)) in enumerate(zip(param_range1, param_range2))
        setproperty!(constants, param1, par1)
        setproperty!(constants, param2, par2)
        time, Vlvmean, lc, plvmean, part = SolveOneFiber(constants, ncycles=ncycles, return_eval=false, print_info=false)

        sens["Plv"][i,:] = plvmean/mmHg 
        sens["Vlv"][i,:] = Vlvmean/ml
    end
    setproperty!(constants, param1, 1)
    setproperty!(constants, param2, 1)
    return sens
end

# sens_αβ = parameter_loop_2(:α, :β, αs, βs)
sens_βϕ = parameter_loop_2(:ϕ, :β, ϕs, βs)

# Postprocessing
Vlv_label = L"V_{lv}\ \mathrm{[ml]}"
Plv_label = L"p_{lv}\ \mathrm{[mmHg]}"
ymax = 200
xmax = 300
#p11 = plot_sens(sens_α, αs, ylabel=Plv_label, title=L"\alpha", arrow=[[200,70],[70,70]], ylims=[0,ymax], xlims=[0, xmax])
p11 = plot_sens(sens_ϕ, ϕs, xlabel=Vlv_label, title=L"\phi", arrow=[[70,200],[70,70]], ylims=[0,ymax], xlims=[0, xmax])
p12 = plot_sens(sens_β, βs,                   title=L"\beta" , arrow=[[70,200],[70,70]], ylims=[0,ymax], xlims=[0, xmax])
p21 = plot_sens(sens_γ, γs, ylabel=Plv_label, xlabel=Vlv_label, title=L"\gamma", arrow=[[200,70],[70,70]], ylims=[0,ymax], xlims=[0, xmax])
p22 = plot_sens(sens_λ, λs, xlabel=Vlv_label, title=L"\lambda", arrow=[[200,70],[70,70]], ylims=[0,ymax], xlims=[0, xmax])

display(plot(p11, p12, p21, p22, layout=(2,2), size=(2*250,2*250), display_type=:gui)) # <- Ensures it is also plotted, not overwritten

display( plot_sens(sens_αβ, αs, ylabel=Plv_label, title=L"\alpha \beta", arrow=[[200,70],[70,70]], ylims=[0,ymax], xlims=[0, xmax]))

display( plot_sens(sens_βϕ, βs, ylabel=Plv_label, title=L"\alpha \beta", arrow=[[200,70],[70,70]], ylims=[0,ymax], xlims=[0, xmax]))
