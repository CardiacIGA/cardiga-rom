using Plots, DataFrames, CSV


# Global struct
struct Postprocessing
    PV_loop::Function
    PV_trace::Function
    Chain::Function
    write_output::Function
end

# Initial constructor
Postprocessing() = Postprocessing(PV_loop, PV_trace, Chain, write_output)

## Function that plots the Pressure-volume loop
function PV_loop(data_points, model_points; data_error_band=Dict(), model_error_band=Dict(), saveFig=false, figureName="PV_loop.png")
    p1 = plot()
    if length(data_error_band) != 0
        # Set color
        if "color" in keys(model_error_band)
            color = model_error_band["color"]
        else
            color = "green"
        end

        XVerts_half1 = data_error_band["Xverts1"]
        XVerts_half2 = data_error_band["Xverts2"]
        YVerts_half1 = data_error_band["Yverts1"]
        YVerts_half2 = data_error_band["Yverts2"]
        for i in range(1,size(XVerts_half1,1))
                plot!([Shape(XVerts_half1[i,:], YVerts_half1[i,:])], label=false, fillcolor=color, linecolor=color, fillalpha=0.4, linealpha=0.) #, c="transparent", fillalpha=1
                plot!([Shape(XVerts_half2[i,:], YVerts_half2[i,:])], label=false, fillcolor=color, linecolor=color, fillalpha=0.4, linealpha=0.)
        end
    end
    if length(model_error_band) != 0
        # Set color
        if "color" in keys(model_error_band)
            color = model_error_band["color"]
        else
            color = "orange"
        end

        XVerts_half1 = model_error_band["Xverts1"]
        XVerts_half2 = model_error_band["Xverts2"]
        YVerts_half1 = model_error_band["Yverts1"]
        YVerts_half2 = model_error_band["Yverts2"]
        for i in range(1,size(XVerts_half1,1))
                plot!([Shape(XVerts_half1[i,:], YVerts_half1[i,:])], label=false, fillcolor=color, linecolor=color, fillalpha=0.4, linealpha=0.) #, c="transparent", fillalpha=1
                plot!([Shape(XVerts_half2[i,:], YVerts_half2[i,:])], label=false, fillcolor=color, linecolor=color, fillalpha=0.4, linealpha=0.)
        end
    end
    plot!(model_points["Vlvs"], model_points["plvs"], linewidth=2, label="One-fiber mean", color=:black, linestyle=:dash, grid=false)
    plot!(data_points["Vlvs"],  data_points["plvs"],  linewidth=2, label="IGA data", color=:black)
    xlabel!(L"$V_{lv}$ [ml]")
    ylabel!(L"$p_{lv}$ [mmHg]")

    figure = p1

    if saveFig
        savefig(figure,figureName)
    end

    return display(figure) 
end

## Function that plots the Pressure-volume traces (over time)
function PV_trace(data_points, model_points; model_error_bound=Dict(), data_error_bound=Dict(), saveFig=false, figureName="PV_trace.png")
    p1 = plot()
    set_model_band_plv = false
    if length(model_error_bound) != 0
        # Set color
        if "color" in keys(model_error_bound)
            color_model = model_error_bound["color"]
        else
            color_model = "orange"
        end
        plot!(model_error_bound["Vlv95_lower"], label=nothing, fill=model_error_bound["Vlv95_upper"], c="transparent", fillcolor=color_model, fillalpha=0.4)
        set_model_band_plv = true
    end

    set_data_band_plv = false
    if length(data_error_bound) != 0
        # Set color
        if "color" in keys(data_error_bound)
            color_data = data_error_bound["color"]
        else
            color_data = "green"
        end
        plot!(data_error_bound["Vlv95_lower"], label=nothing, fill=data_error_bound["Vlv95_upper"], c="transparent", fillcolor=color_data, fillalpha=0.4)
        set_data_band_plv = true
    end

    plot!(model_points["Vlvs"], linewidth=2, c=:black, linestyle=:dash, label="Mean")
    plot!(data_points["Vlvs"], linewidth=2, label="IGA", c=:black)
    ylabel!(L"$V^{\mathrm{lv}}$ [ml]")

    p2 = plot()
    if set_model_band_plv
        plot!(model_error_bound["plv95_lower"], label=nothing, fill=model_error_bound["plv95_upper"], c="transparent", fillcolor=color_model, fillalpha=0.4)
    end
    if set_data_band_plv
        plot!(data_error_bound["plv95_lower"], label=nothing, fill=data_error_bound["plv95_upper"], c="transparent", fillcolor=color_data, fillalpha=0.4)
    end

    plot!(model_points["plvs"], linewidth=2, c=:black, linestyle=:dash, label="Mean")
    plot!(data_points["plvs"], linewidth=2, label="IGA", c=:black)
    ylabel!(L"$p^{\mathrm{lv}}$ [mmHg]")
    xlabel!(L"\mathrm{Time\ idx}\ [-]")

    figure = plot(p1, p2, layout=(2,1))

    if saveFig
        savefig(figure,figureName)
    end

    return display(figure)
end


function Chain(samples, priors; saveFig=false, figureName="Chains.png")

    # Unpack
    #αs = samples.α
    βs = samples.β
    γs = samples.γ
    λs = samples.λ
    ϕs = samples.ϕ

    #αprior = priors.α
    βprior = priors.β
    γprior = priors.γ
    λprior = priors.λ
    ϕprior = priors.ϕ

    # Number of uniform samples
    nsmpls  = 100
    
    #αarray  = collect(range(start=minimum(αs), stop=maximum(αs), length=nsmpls))
    βarray  = collect(range(start=minimum(βs), stop=maximum(βs), length=nsmpls))
    γarray  = collect(range(start=minimum(γs), stop=maximum(γs), length=nsmpls))
    λarray  = collect(range(start=minimum(λs), stop=maximum(λs), length=nsmpls))
    ϕarray  = collect(range(start=minimum(ϕs), stop=maximum(ϕs), length=nsmpls))

    #αlogprior   = logpdf.(αprior, αarray)
    βlogprior   = logpdf.(βprior, βarray)
    γlogprior   = logpdf.(γprior, γarray)
    λlogprior   = logpdf.(λprior, λarray)
    ϕlogprior   = logpdf.(ϕprior, ϕarray)

    # Mean and standard deviations
    #αmean = mean(αs)
    βmean = mean(βs)
    γmean = mean(γs)
    λmean = mean(λs)
    ϕmean = mean(ϕs)

    # Distributions and chains/walker samples
    #p11 = plot(αs, ylabel=L"\alpha", xlabel="Samples", label=false) # Plot the walker, sampled values vs iterations/samples
    p11 = plot(ϕs, ylabel=L"\phi", xlabel="Samples", label=false)
    p21 = plot(βs, ylabel=L"\beta", xlabel="Samples", label=false)  # Plot the walker, sampled values vs iterations/samples
    p31 = plot(γs, ylabel=L"\gamma", xlabel="Samples", label=false)
    p41 = plot(λs, ylabel=L"\lambda", xlabel="Samples", label=false)

    # p12 = density(αs, xlabel=L"\alpha\ (\mathrm{Geom})", ylabel="Density", fill=(0, .5,:gray), linewidth=3, label="Posterior") # Plots the density of sampled values
    # plot!(αarray, exp.(αlogprior), color=:green, linewidth=1, linestyle=:dot, label="Prior")
    # vline!([αmean], linewidth=3, color=:red, label="Mean")
    p12 = density(ϕs, xlabel=L"\phi\ (\mathrm{Geom})", ylabel="Density", fill=(0, .5,:gray), linewidth=3, label="Posterior") # Plots the density of sampled values
    plot!(ϕarray, exp.(ϕlogprior), color=:green, linewidth=1, linestyle=:dot, label="Prior")
    vline!([ϕmean], linewidth=3, color=:red, label="Mean")
    p22 = density(βs, xlabel=L"\beta\ (\mathrm{Geom})", ylabel="Density", fill=(0, .5,:gray), linewidth=3, label="Posterior") # Plots the density of sampled values
    plot!(βarray, exp.(βlogprior), color=:green, linewidth=1, linestyle=:dot, label="Prior")
    vline!([βmean], linewidth=3, color=:red, label="Mean")
    p32 = density(γs, xlabel=L"\gamma\ (T_{p0})", ylabel="Density", fill=(0, .5,:gray), linewidth=3, label="Posterior") # Plots the density of sampled values
    plot!(γarray, exp.(γlogprior), color=:green, linewidth=1, linestyle=:dot, label="Prior")
    vline!([γmean], linewidth=3, color=:red, label="Mean")
    p42 = density(λs, xlabel=L"\lambda\ (c_p)", ylabel="Density", fill=(0, .5,:gray), linewidth=3, label="Posterior") # Plots the density of sampled values
    plot!(λarray, exp.(λlogprior), color=:green, linewidth=1, linestyle=:dot, label="Prior")
    vline!([λmean], linewidth=3, color=:red, label="Mean")

    figure = plot(p11, p12, p21, p22, p31, p32, p41, p42, layout=(4,2), size=(2*250,4*250), display_type=:gui) # <- Ensures it is also plotted, not overwritten

    if saveFig
        savefig(figure,figureName)
    end

    return display(figure)

    

end

function write_output(samples; filename="output_file.csv")
    CSV.write(filename, DataFrame([samples]))
    return
end