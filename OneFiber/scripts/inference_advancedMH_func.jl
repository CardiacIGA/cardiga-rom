# using AdvancedMH, MCMCChains
# using OneFiber, Plots, StatsPlots, LaTeXStrings, AxisArrays
# using DelimitedFiles, Distributions, LinearAlgebra

# Include adaptive function
# include("adaptive_inference.jl")
# include("Postprocessing.jl")
using OneFiber
using DelimitedFiles, Distributions, LinearAlgebra


function Inference_AdvMH(args)

    point = args[1]

    inputFolder  = "input/"
    outputFolder = "output/"

    # Initialize input values
    constants = Constants()
    (; ms, ml, kPa, mmHg, tcycle, δt) = constants

    # Input settings samplers
    nsamples_adaptMH       = 100_000#100_000
    resets_everyn_adaptMH  = 25_000#10_000
    target_accept          = 0.234

    nsamples_MH   = 25_000
    nburnin_MH    =  5000 

    ncycles   = 6
    idx_range = ((ncycles-1)*Int(tcycle/δt) + 1):(ncycles*Int(tcycle/δt))


    ## Load the data (OneFiber REF)
    # data_matrix = readdlm("data/results_verification.csv", ',')
    # data = Dict(i[1] => i[2:end] for i in eachcol(data_matrix))

    ## load the data (IGA)
    #point       = "1" 
    dimension   = Dict("1" => "C043_H096", "2"  => "C075_H045", "3"  => "C113_H00", "4"  => "C075_H124", "5"  => "C114_H06",
                       "6"  => "C154_H00", "7"  => "C109_H15", "8"  => "C153_H075", "9"  => "C195_H00", "10" => "C10_H10")      
    filenameIGA = string(inputFolder, "results_IGA_GtypeP", point, "_", dimension[point], ".csv") #
    # filenameIGA = "data/result_IGA.csv"
    data_matrix = readdlm(filenameIGA, ',')
    data = Dict(i[1] => i[2:end] for i in eachcol(data_matrix))
    data["Vlvs"] /= ml
    data["Vlvs"] = data["Vlvs"][idx_range] #[1:400]# 
    data["plvs"] = data["plvs"][idx_range]  #[1:400] #
    data_input   = [data["Vlvs"]/ml data["plvs"]/mmHg] # Convert to ml and mmHg Units
    data_points  = Dict("Vlvs" => data_input[:,1], "plvs" => data_input[:,2])

    ## Set-up the priors
    #αprior = Normal(1, 0.3) # Geometry scaling factor
    βprior = Normal(1, 0.1) # Geometry scaling factor
    γprior = Normal(1, 0.1)#0.3) # Tp0 scaling factor
    λprior = Normal(1, 0.1) # cp scaling factor
    ϕprior = Normal(1, 0.1) # cp scaling factor
    #ωprior = Normal(1, 0.3) # cp scaling factor
    #priors = (α=αprior, β=βprior, γ=γprior, λ=λprior, ϕ=ϕprior)
    priors = (β=βprior, γ=γprior, λ=λprior, ϕ=ϕprior)
    #priors = (β=βprior, γ=γprior, λ=λprior, ω=ωprior)

    σ_Vlv = 0.03
    σ_Plv = 0.05

    σnoise_Vlv = 0.5*(120 - 44)*σ_Vlv # 0.05 # Noise in [ml]
    σnoise_Plv = 2.5  #0.5*(120 -  0)*σ_Plv # Noise in [mmHg]

    # Set noise decrease/increase instances
    valve_points = return_valve_points(data_input[:,2], data_input[:,1], cycle=ncycles, return_idx=true)
    t1 = Int(valve_points[2,1])
    t2 = (Int(valve_points[3,1]), Int(valve_points[4,1]))
    t3 = Int(valve_points[1,1])
    # Create heavy-sided noise function
    fslope(x; k=10, t=0, p=1) = p*tanh( k*(x - t) ) + 1
    heavys1(x; k=10, t1=100) = 0.5*( fslope(x, k=k, t=-300, p=1) + fslope(x, k=k, t=t1, p=-1) ) - 1
    heavys2(x; k=10, t1=175, t2=250) = 0.5*( fslope(x, k=k, t=t1, p=1) + fslope(x, k=k, t=t2, p=-1) ) - 1
    heavys3(x; k=10, t1=300) = 0.5*fslope(x, k=k, t=t1, p=1)
    fnoise(x; k=10, t1=100, t2=(180,300), t3=360) = heavys1(x; k=k, t1=t1) + heavys2(x; k=k, t1=t2[1], t2=t2[2]) + heavys3(x; k=k, t1=t3)
    # fnoise(x; k=10, t1=(150,300), t2=(450,500)) = heavys(x; k=k, t1=t1[1], t2=t1[2]) + heavys(x; k=k, t1=t2[1], t2=t2[2])

    σslope = 0.5
    σfrac  = 0.78 #0.6
    σnoise_Vlv_arr = ( σfrac*σnoise_Vlv ) .* fnoise.(range(1,400), k=σslope, t1=t1, t2=t2, t3=t3) .+ (1-σfrac)*σnoise_Vlv
    Σnoise_Vlv_mat = Diagonal(σnoise_Vlv_arr) #σnoise_Vlv_arr .* σnoise_Vlv_arr'

    l     = 1 # length scale
    ts_05 = 399 #idx_range[end-1]
    ts    = abs.(collect(range( -ts_05, ts_05 )))
    Σtime = stack([ts[(ts_05+2-i):(end+1-i)] for i in range(1,size(ts[ts_05:end],1)-1)],dims=2)
    Σplv  = Symmetric( σnoise_Plv^2 * exp.( -δt/( tcycle - δt ) .* l * Σtime ) )
    ΣVlv  = Symmetric( Σnoise_Vlv_mat * exp.( -δt/( tcycle - δt ) .* l * Σtime ) * Σnoise_Vlv_mat )

    ## Run the adaptive metropolis
    println("Running P", point)
    out_adaptive = run_RobustAdaptiveMetropolis(data_input, priors, constants; ncycles=ncycles, 
                                        target_accept=target_accept, nsamples=nsamples_adaptMH, restart_every_nth=resets_everyn_adaptMH, 
                                        ΣVlv=ΣVlv, Σplv=Σplv)
    println(out_adaptive.M)
    println(vec(mean(out_adaptive.chain, dims=1)))


    # Point 1 6th cycle, init=...
    # Σcov = ...

    # Point 2 6th cycle, init=...
    # Σcov = ...

    # Point 3 6th cycle, init=...
    # Σcov = ...

    # Point 4 6th cycle, init=...
    # Σcov = ...

    # Point 5 6th cycle, init=...
    # Σcov = ...

    # Point 6 6th cycle, init=...
    # Σcov = ...

    # Point 7 6th cycle, init=...
    # Σcov = ...

    # Point 8 6th cycle, init=...
    # Σcov = ...

    # Point 9 6th cycle, init=...
    # Σcov = ...

    # Point 10 6th cycle, init=...
    # Σcov = ...


    Σcov       = out_adaptive.M 
    samples    = Samples(true,                                                   # Set true if samples should be stores
                        Array{Float64}(undef, nsamples_MH, size(data["Vlvs"],1)),  # Initialize stored Vlv array
                        Array{Float64}(undef, nsamples_MH, size(data["plvs"],1)),  # Initialize stored plv array
                        1,                                                      # Starting index 
                        nburnin_MH)                                             # Number of burn-in samples
    LogP_model = return_LogP_Onefiber(data_input, priors, constants; ncycles=ncycles, ΣVlv=ΣVlv, Σplv=Σplv, samples=samples)
    model      = DensityModel( LogP_model )
    initial    = vec(mean(out_adaptive.chain, dims=1)) # Ref: [0.817, 4.496, 0.879, 1.084]
    sampler    = RWMH(MvNormal(zeros(size(Σcov,1)), Σcov))
    chain      = sample(model, sampler, nsamples_MH, discard_initial=nburnin_MH, param_names = ["β", "γ", "λ", "ϕ"], initial_params = initial, chain_type = Chains)


    ## Post-processing
    βs  = Array(getproperty(get(chain, Symbol("β")), Symbol("β")))
    γs  = Array(getproperty(get(chain, Symbol("γ")), Symbol("γ")))
    λs  = Array(getproperty(get(chain, Symbol("λ")), Symbol("λ")))
    ϕs  = Array(getproperty(get(chain, Symbol("ϕ")), Symbol("ϕ")))

    chain_samples = (β=βs, γ=γs, λ=λs, ϕ=ϕs)

    βarray  = collect(range(start=minimum(βs), stop=maximum(βs), length=100))
    γarray  = collect(range(start=minimum(γs), stop=maximum(γs), length=100))
    λarray  = collect(range(start=minimum(λs), stop=maximum(λs), length=100))
    ϕarray  = collect(range(start=minimum(ϕs), stop=maximum(ϕs), length=100))


    # αlogprior   = logpdf.(αprior, αarray)
    βlogprior   = logpdf.(βprior, βarray)
    γlogprior   = logpdf.(γprior, γarray)
    λlogprior   = logpdf.(λprior, λarray)
    ϕlogprior   = logpdf.(ϕprior, ϕarray)

    # Run model with average values ← Final value
    constants.β = mean(βs)
    constants.γ = mean(γs)
    constants.λ = mean(λs)
    constants.ϕ = mean(ϕs)
    time, Vlvmean, lc, plvmean, part = SolveOneFiber(constants, ncycles=ncycles, return_eval=false, print_info=false) #, initial=initial

    Vlvmean = Vlvmean[idx_range]
    plvmean = plvmean[idx_range]
    model_points = Dict("Vlvs" => Vlvmean/ml, "plvs" => plvmean/mmHg)

    # Sample noise
    Vlv_noise = MvNormal(zeros(size(ΣVlv,1)), ΣVlv*(ml^2)) # Pull back to [m] units #Normal(0., σnoise_Vlv*ml)
    Vlv_noise_sample = rand(Vlv_noise, size(samples.Vlv,1)) # Draw sufficient amount of sampled points #size(samples.Vlv,1))
    Plv_noise = MvNormal(zeros(size(Σplv,1)), Σplv*(mmHg^2)) # Pull back to [Pa] units #Normal(0., σnoise_Plv*mmHg)
    Plv_noise_sample = rand(Plv_noise, size(samples.Plv,1)) #size(samples.Plv,1))

    Vlv_samples = samples.Vlv .+ Vlv_noise_sample'
    Plv_samples = samples.Plv .+ Plv_noise_sample'


    # Extract 95% deviations (We assume normal distribution)
    Vlv95_upper = vec(mean(Vlv_samples, dims=1) + 2*std(Vlv_samples, dims=1)) #vec(maximum(samples.Vlv, dims=1)) #
    Vlv95_lower = vec(mean(Vlv_samples, dims=1) - 2*std(Vlv_samples, dims=1)) #vec(minimum(samples.Vlv, dims=1)) #

    plv95_upper = vec(mean(Plv_samples, dims=1) + 2*std(Plv_samples, dims=1)) #vec(maximum(samples.Plv, dims=1)) #
    plv95_lower = vec(mean(Plv_samples, dims=1) - 2*std(Plv_samples, dims=1)) #vec(minimum(samples.Plv, dims=1)) #

    Vlv95_data_upper = data_points["Vlvs"] .+ 2*σnoise_Vlv_arr #2*σnoise_Vlv 
    Vlv95_data_lower = data_points["Vlvs"] .- 2*σnoise_Vlv_arr #2*σnoise_Vlv #vec(minimum(samples.Vlv, dims=1)) #

    plv95_data_upper = data_points["plvs"] .+ 2*σnoise_Plv #vec(maximum(samples.Plv, dims=1)) #
    plv95_data_lower = data_points["plvs"] .- 2*σnoise_Plv #vec(minimum(samples.Plv, dims=1)) #

    # Extract 95% confidence interval for PV-loop
    function compute_normal_PV(P,V)
        normal  = diff([V P],dims=1) * [0  1; -1  0] # Pointing outward of the PV loop, divide by units, to circumvent underflow
        normal ./= sqrt.( sum( normal.^2, dims=2) ) 
        normal  = [normal; normal[end,:]']
        
        # Perform averaging for smoothing
        dnormals = 0.5*(normal[1:end-1,:] + normal[2:end,:])#diff(normal, dims=1)
        normal = [dnormals; dnormals[end,:]']
        return normal
    end
    normal_sim  = compute_normal_PV(plvmean/mmHg, Vlvmean/ml)
    normal_data = compute_normal_PV(data["plvs"]/mmHg, data["Vlvs"]/ml)


    ## TODO change normal computation for IGA results
    PV95_noise       =  (normal_data.*[σnoise_Vlv_arr σnoise_Plv*ones(size(σnoise_Vlv_arr,1))])#(normal.*[σnoise_Vlv σnoise_Plv])
    PV95_noise_upper =  [data_points["Vlvs"] data_points["plvs"]] .+ 2*PV95_noise
    PV95_noise_lower =  [data_points["Vlvs"] data_points["plvs"]] .- 2*PV95_noise

    PV95_sim       =  (normal_sim.*[std(Vlv_samples, dims=1); std(Plv_samples, dims=1)]')
    PV95_sim_upper = ( [Vlvmean plvmean] .+ 2*PV95_sim ) ./ [ml mmHg]
    PV95_sim_lower = ( [Vlvmean plvmean] .- 2*PV95_sim ) ./ [ml mmHg]


    # Create Tet polygons
    XVerts_sim_half1 = [PV95_sim_upper[1:(end-1),1] PV95_sim_upper[2:end,1] PV95_sim_lower[1:(end-1),1]]
    YVerts_sim_half1 = [PV95_sim_upper[1:(end-1),2] PV95_sim_upper[2:end,2] PV95_sim_lower[1:(end-1),2]]

    XVerts_sim_half2 = [PV95_sim_lower[1:(end-1),1] PV95_sim_lower[2:end,1] PV95_sim_upper[2:(end),1]]
    YVerts_sim_half2 = [PV95_sim_lower[1:(end-1),2] PV95_sim_lower[2:end,2] PV95_sim_upper[2:(end),2]]

    XVerts_noise_half1 = [PV95_noise_upper[1:(end-1),1] PV95_noise_upper[2:end,1] PV95_noise_lower[1:(end-1),1]]
    YVerts_noise_half1 = [PV95_noise_upper[1:(end-1),2] PV95_noise_upper[2:end,2] PV95_noise_lower[1:(end-1),2]]

    XVerts_noise_half2 = [PV95_noise_lower[1:(end-1),1] PV95_noise_lower[2:end,1] PV95_noise_upper[2:(end),1]]
    YVerts_noise_half2 = [PV95_noise_lower[1:(end-1),2] PV95_noise_lower[2:end,2] PV95_noise_upper[2:(end),2]]




    # Combin into dictionaries
    data_error_band  = Dict("Xverts1" => XVerts_noise_half1, "Yverts1" => YVerts_noise_half1, "Xverts2" => XVerts_noise_half2, "Yverts2" => YVerts_noise_half2)
    model_error_band = Dict("Xverts1" => XVerts_sim_half1  , "Yverts1" => YVerts_sim_half1  , "Xverts2" => XVerts_sim_half2  , "Yverts2" => YVerts_sim_half2)

    data_error_bound  = Dict("Vlv95_lower" => Vlv95_data_lower, "Vlv95_upper" => Vlv95_data_upper, "plv95_lower" => plv95_data_lower, "plv95_upper" => plv95_data_upper)
    model_error_bound = Dict("Vlv95_lower" => Vlv95_lower/ml, "Vlv95_upper" => Vlv95_upper/ml, "plv95_lower" => plv95_lower/mmHg, "plv95_upper" => plv95_upper/mmHg)



    ## Post processing
    post = Postprocessing()
    post.PV_loop(data_points, model_points;     data_error_band  = data_error_band , model_error_band  = model_error_band, 
                 saveFig=true, figureName=string(outputFolder, "figures/", "PV_loop", point, "_", dimension[point], ".png"))

    post.PV_trace(data_points, model_points;    data_error_bound = data_error_bound, model_error_bound = model_error_bound,
                  saveFig=true, figureName=string(outputFolder, "figures/", "PV_trace", point, "_", dimension[point], ".png"))

    post.Chain( chain_samples, priors, saveFig=true, 
                  figureName=string(outputFolder, "figures/", "Chain", point, "_", dimension[point], ".png"))


    # Write calibrated parameter values incl. uncertainty to output file
    chain_samples = (# Parameters
                    beta=βs, gamma=γs, lambda=λs, phi=ϕs, 

                    # Pressure-Volume traces
                    Vlv_mean    = mean(Vlv_samples, dims=1),
                    Vlv_95upper = Vlv95_upper,
                    Vlv_95lower = Vlv95_lower,
                    Plv_mean    = mean(Plv_samples, dims=1),
                    Plv_95upper = plv95_upper,
                    Plv_95lower = plv95_lower,

                    # Pressure-Volume loop
                    PV_mean    = PV95_sim,     
                    PV_95upper = PV95_sim_upper,
                    PV_95lower = PV95_sim_lower,
                    PV95_sim   = PV95_sim,

                    # Pressure-Volume data points
                    Vlv_mean_data   = data_points["Vlvs"],
                    Plv_mean_data   = data_points["plvs"],
                    PV95_noise_data = PV95_noise,

                    # Cycle number which was inferred
                    ncycles = ncycles)


    post.write_output(chain_samples; filename=string(outputFolder, "data/", "parameters_P", point, ".csv"))
end

Inference_AdvMH(ARGS)