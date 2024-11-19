# using OneFiber, Plots, StatsPlots, LaTeXStrings, AxisArrays
# using DelimitedFiles, Random, Distributions, LinearAlgebra, KernelDensity
# using RobustAdaptiveMetropolisSampler, VegaLite, AdaptiveMCMC

using AxisArrays
using Random, Distributions, LinearAlgebra, KernelDensity
using RobustAdaptiveMetropolisSampler 


mutable struct Samples
    Store::Bool
    Vlv::Array{Float64}
    Plv::Array{Float64}
    idx::Int64
    burnin::Int64
end

# Initial constructor
Samples() = Samples(false, [1.], [1.], 1, 1)



## Define log-function
function return_LogP_Onefiber(qois, priors, constants; ncycles=1, initial = Dict{String, Float64}("Vlv" => 0.044, "lc" => 1.5, "plv" => 0., "part" => 11_499.9875), cycle_idx_range=false,
                                Σplv=Diagonal([(0.1)^2 for prior in priors]), ΣVlv=Diagonal([(0.1)^2 for prior in priors]), samples=Samples())

    inv_ΣVlv = inv(ΣVlv)
    inv_Σplv = inv(Σplv)

    # Retrieve units
    (; ms, ml, kPa, mmHg, tcycle, δt) = constants
    

    idx_range = ((ncycles-1)*Int(tcycle/δt) + 1):(ncycles*Int(tcycle/δt))
    
    let qois=qois, priors=priors, constants=constants, samples=samples, ncycles=ncycles, initial=initial,
        inv_Σplv=inv_Σplv, inv_ΣVlv=inv_ΣVlv, # Inverse of covariance matrices of the data noise
        ml=ml, mmHg=mmHg, idx_range=idx_range, cycle_idx_range=cycle_idx_range   # Useful unit conversions and time measures

        function log_p(var)
            # Unpack the stochastic variables
            #α, β, γ, λ, ϕ = var
            β, γ, λ, ϕ = var
            #β, γ, λ, ω = var
            
            # Return Inf for log-density if we have a negative parameter value or the β-parameter is too small (causing overflow in the model -> x.^(Inf) for example)
            if any(var .< 0.03) #| ( β < 0.03 )
                return -Inf # Return minus infinite posterior value
            end

            # Cap the input, otherwise gets stuck 
            ##TODO Convert to adding -∞ to l_dens
            #constants.α = max(α, 0.1) # Geometric parameter
            constants.β = β #min(max(β, 0.1), 5.0) # Geometric parameter
            constants.γ = γ #min(max(γ, 0.1), 8.0) # Mechanics parameter
            constants.λ = λ #min(max(λ, 0.1), 5.0) # Mechanics parameter
            constants.ϕ = ϕ #min(max(ϕ, 0.1), 5.0) # Mechanics parameter
            #constants.ω = max(ω, 0.1) # Mechanics parameter

            # println("β: ", constants.β)
            # println("γ: ", constants.γ)
            # println("λ: ", constants.λ)
            # println("ϕ: ", constants.ϕ)

            

            # Solve the time dependent problem
            time, Vlv, lc, plv, part = SolveOneFiber(constants, ncycles=ncycles, initial=initial, return_eval=false, print_info=false) #, initial=initial
            
            # println(Vlv[1])
            # Return Inf for log-density if we have a diverged solution
            if Vlv[1] == Inf
                return -Inf # Return minus infinite posterior value
            end
            
            if cycle_idx_range 
                valve_idx = return_valve_points_new(plv, Vlv, cycle=ncycles, return_idx=true)
                idx_range = (Int(valve_idx[1,1]) - Int(tcycle/δt)):(Int(valve_idx[1,1])-1) # Neglect last index
            end

            # Store the sampled Vlv and Plv if specified: THIS IS NOT SAFE, ALSO INCLUDES REJECTED SAMPLES!
            if samples.Store && (samples.idx > samples.burnin)
                samples.Vlv[samples.idx-samples.burnin,:] = Vlv[idx_range]
                samples.Plv[samples.idx-samples.burnin,:] = plv[idx_range]
            end
            samples.idx += 1

            # Retrieve logpdf value
            #α_logprior = logpdf( priors.α, α)
            β_logprior = logpdf( priors.β, β)
            γ_logprior = logpdf( priors.γ, γ)
            λ_logprior = logpdf( priors.λ, λ)
            ϕ_logprior = logpdf( priors.ϕ, ϕ)
            #ω_logprior = logpdf( priors.ω, ω)

            # Set up the posterior logpdf
            # l_dens  = -0.5*sum((qois[:,1] .- Vlv[idx_range]/ml  ).^2 ./ (σx.^2))
            # l_dens -=  0.5*sum((qois[:,2] .- plv[idx_range]/mmHg).^2 ./ (σy.^2))

            l_dens  = -0.5*sum(  (qois[:,1] .- Vlv[idx_range]/ml  )' * inv_ΣVlv * (qois[:,1] .- Vlv[idx_range]/ml  ))
            l_dens  -=  0.5*sum(  (qois[:,2] .- plv[idx_range]/mmHg)' * inv_Σplv * (qois[:,2] .- plv[idx_range]/mmHg))

            
    
            # Prevent underflow by log-sum trick
            l_total = l_dens + β_logprior + γ_logprior + λ_logprior + ϕ_logprior #+ ω_logprior #
            # println(l_total)
            #print(l_dens)

            # println("LogD-tot: ", l_total)
            # println("Log-dens: ", l_dens)
            # println("βlog    : ", β_logprior," β : ", constants.β)
            # println("γlog,   : ", γ_logprior," γ : ", constants.γ)
            # println("λlog,   : ", λ_logprior," λ : ", constants.λ)
            # println("ϕlog,   : ", ϕ_logprior," ϕ : ", constants.ϕ)
            return l_total
        end
    end
end

function run_RobustAdaptiveMetropolis(data, priors, constants; initial = Dict{String, Float64}("Vlv" => 0.044, "lc" => 1.5, "plv" => 0., "part" => 11_499.9875), cycle_idx_range=false,
                                            target_accept=0.234, nsamples=10_000, restart_every_nth = nsamples, ncycles=1, Σplv=Diagonal([(0.1)^2 for i in range(1,size(data,1))]), ΣVlv=Diagonal([(0.1)^2 for i in range(1,size(data,1))]), samples=Samples())

    # Load units
    #(; ms, ml, kPa, mmHg) = constants

    # Initialize the log-posterior model
    model   = return_LogP_Onefiber(data, priors, constants; ncycles=ncycles, initial=initial, Σplv=Σplv, ΣVlv=ΣVlv, samples=samples, cycle_idx_range=cycle_idx_range)

    # Set initial values
    σ0      = [std(prior)^2 for prior in priors] 
    Σ0      = Diagonal(σ0) 
    initial = vec(ones(size(Σ0,1)))
 
    for i in range(1,div(nsamples, restart_every_nth))

        # Run adaptive Markov chain
        global out_adaptive = RAM_sample(
            model,                 # log target function
            initial,               # Initial values
            Σ0,                    # Scaling factor 
            restart_every_nth,     # Number of runs
            opt_α = target_accept,  # Optimal acceptance rate solver aims for
            output_log_probability_x = true
            )

        println("Lps  :", out_adaptive.log_probabilities_x[end-10:end])
        println("Means:   ", mean(out_adaptive.chain, dims=1))
        println("Cov-Mat: ", out_adaptive.M)
        # Set values for next iteration    
        initial = out_adaptive.chain[end,:] # Use last values of the chain again
        Σ0      = copy(out_adaptive.M) # Store covariance matrix and resubmit it as initial if sampling is restarted
    end

    return out_adaptive
end

# using AdaptiveMCMC
# function run_AdaptiveMetropolis(data, priors, constants; target_accept=0.234, nsamples=10_000, restart_every_nth = nsamples, ncycles=1, Σplv=Diagonal([(0.1)^2 for i in range(1,size(data,1))]), ΣVlv=Diagonal([(0.1)^2 for i in range(1,size(data,1))]), samples=Samples())

#     # Load units
#     #(; ms, ml, kPa, mmHg) = constants

#     # Initialize the log-posterior model
#     model   = return_LogP_Onefiber(data, priors, constants; ncycles=ncycles, Σplv=Σplv, ΣVlv=ΣVlv, samples=samples)

#     # Set initial values
#     σ0      = [std(prior)^2 for prior in priors] 
#     Σ0      = Diagonal(σ0) 
#     initial = vec(ones(size(Σ0,1)))
#     #sampler = WMH(MvNormal(zeros(size(Σ0,1)), Σ0)) 
#     algorithm = :ram
#     burn_in = 0


#     for i in range(1,div(nsamples, restart_every_nth))

#         # Run adaptive Markov chain
#         global out_adaptive = adaptive_rwm(
#             initial,               # Initial values
#             model,                 # log target function
#             restart_every_nth,     # Number of runs
#             b = burn_in,
#             acc_sw = target_accept, # Optimal acceptance rate solver aims for
#             #thin=20,
#             #q=sampler,
#             algorithm=algorithm,   # Type of adaptive algorithm used: 'ram', 'am', 'asm', 'aswam', 'rwm'
#             progress=true)

#         # Set values for next iteration    
#         # initial = out_adaptive.chain[end,:] # Use last values of the chain again
#         # Σ0      = copy(out_adaptive.M) # Store covariance matrix and resubmit it as initial if sampling is restarted
#     end

#     return out_adaptive
# end





# main = false
# if main
# ## Load units
# constants = Constants()
# (; ms, ml, kPa, mmHg) = constants

# ## load the data (IGA)
# data_matrix = readdlm("data/result_IGA.csv", ',')
# data = Dict(i[1] => i[2:end] for i in eachcol(data_matrix))
# data["Vlvs"] /= ml
# data["Vlvs"] = data["Vlvs"][1:401]#[2001:2400] 
# data["plvs"] = data["plvs"][1:401] #[2001:2400]  

# ## Set-up the priors
# αprior = Normal(1, 0.3) # Geometry scaling factor
# βprior = Normal(1, 0.3) # Geometry scaling factor
# γprior = Normal(1, 0.3) # Tp0 scaling factor
# λprior = Normal(1, 0.3) # cp scaling factor

# priors = (α=αprior, β=βprior, γ=γprior, λ=λprior)


# ## Run the adaptive metropolis
# # out_adaptive = run_AdaptiveMetropolis(data, priors, constants, target_accept=0.234, nsamples=100_000, restart_every_nth=25_000, σx=0.1, σy=0.1)



# ## Alternative module
# # using AdaptiveMCMC
# # out = adaptive_rwm(zeros(2), log_p, 10_000; algorithm=:am) # :ram, :am, :asm, :aswam, :rwm
# # using AdvancedMH
# # M = Cov =  [1.37883e-5  2.11924e-6  -4.25427e-7  -8.77065e-6;
# #             2.11924e-6  2.87427e-6   6.30631e-7   4.35441e-7;
# #             -4.25427e-7  6.30631e-7   0.00082905  -0.00017462;
# #             -8.77065e-6  4.35441e-7  -0.00017462   4.40288e-5]
# # model   = DensityModel( return_LogP_Onefiber([data["Vlvs"]/ml data["plvs"]/mmHg], priors, constants) )
# # sampler = RWMH(MvNormal(ones(4), out_adaptive.M))
# # initial = vec(mean(out_adaptive.chain, dims=1))
# # chain   = sample(model, sampler, 10_000, param_names = ["α", "β", "γ", "λ"], initial_params = initial, chain_type = Chain)

# ## Post-processing
# αs  = out_adaptive.chain[:,1]
# βs  = out_adaptive.chain[:,2]
# γs  = out_adaptive.chain[:,3]
# λs  = out_adaptive.chain[:,4]
# #lps = get(chain, :lp).lp

# αarray  = collect(range(start=minimum(αs), stop=maximum(αs), length=100))
# βarray  = collect(range(start=minimum(βs), stop=maximum(βs), length=100))
# γarray  = collect(range(start=minimum(γs), stop=maximum(γs), length=100))
# λarray  = collect(range(start=minimum(λs), stop=maximum(λs), length=100))

# αlogprior   = logpdf.(αprior, αarray)
# βlogprior   = logpdf.(βprior, βarray)
# γlogprior   = logpdf.(γprior, γarray)
# λlogprior   = logpdf.(λprior, λarray)

# dens_max(vval) = kde(vec(vval)).x[ findmax(kde(vec(vval)).density)[2] ]

# # Run model with average values ← Final value
# constants.α = dens_max(αs)
# constants.β = dens_max(βs)
# constants.γ = dens_max(γs)
# constants.λ = dens_max(λs)
# time, Vlvmean, lc, plvmean, part = SolveOneFiber(constants, ncycles=1, return_eval=false, print_info=false) #, initial=initial

# # PV-loop
# plot(Vlvmean/ml, plvmean/mmHg, linewidth=2, label="Mean", color=:black, linestyle=:dash)
# display(plot!(data["Vlvs"]/ml, data["plvs"]/mmHg, linewidth=2, label="IGA", color=:black))
# xlabel!(L"$V_{lv}$ [ml]")
# ylabel!(L"$p_{lv}$ [mmHg]")

# # Distributions and chains/walker samples
# p11 = plot(αs, ylabel=L"\alpha", xlabel="Samples", label=false) # Plot the walker, sampled values vs iterations/samples
# p21 = plot(βs, ylabel=L"\beta", xlabel="Samples", label=false)  # Plot the walker, sampled values vs iterations/samples
# p31 = plot(γs, ylabel=L"\gamma", xlabel="Samples", label=false)
# p41 = plot(λs, ylabel=L"\lambda", xlabel="Samples", label=false)

# p12 = density(αs, xlabel=L"\alpha\ (\mathrm{Geom})", ylabel="Density", fill=(0, .5,:gray), linewidth=3, label="Posterior") # Plots the density of sampled values
# plot!(αarray, exp.(αlogprior), color=:green, linewidth=1, linestyle=:dot, label="Prior")
# vline!([mean(αs)], linewidth=3, color=:red, label="Mean")
# p22 = density(βs, xlabel=L"\beta\ (\mathrm{Geom})", ylabel="Density", fill=(0, .5,:gray), linewidth=3, label="Posterior") # Plots the density of sampled values
# plot!(βarray, exp.(βlogprior), color=:green, linewidth=1, linestyle=:dot, label="Prior")
# vline!([mean(βs)], linewidth=3, color=:red, label="Mean")
# p32 = density(γs, xlabel=L"\gamma\ (T_{p0})", ylabel="Density", fill=(0, .5,:gray), linewidth=3, label="Posterior") # Plots the density of sampled values
# plot!(γarray, exp.(γlogprior), color=:green, linewidth=1, linestyle=:dot, label="Prior")
# vline!([mean(γs)], linewidth=3, color=:red, label="Mean")
# p42 = density(λs, xlabel=L"\lambda\ (c_p)", ylabel="Density", fill=(0, .5,:gray), linewidth=3, label="Posterior") # Plots the density of sampled values
# plot!(λarray, exp.(λlogprior), color=:green, linewidth=1, linestyle=:dot, label="Prior")
# vline!([mean(λs)], linewidth=3, color=:red, label="Mean")

# display(plot(p11, p12, p21, p22, p31, p32, p41, p42, layout=(5,2), size=(2*250,4*250), display_type=:gui)) # <- Ensures it is also plotted, not overwritten
# # end

# Load units
# constants = Constants()
# (; ms, ml, kPa, mmHg) = constants

# # load the data (IGA)
# data_matrix = readdlm("data/result_IGA.csv", ',')
# data = Dict(i[1] => i[2:end] for i in eachcol(data_matrix))
# data["Vlvs"] /= ml
# data["Vlvs"] = data["Vlvs"][1:401]#[2001:2400] 
# data["plvs"] = data["plvs"][1:401] #[2001:2400]  

# # load the data (OneFiber)
# # data_matrix = readdlm("data/results_verification.csv", ',')
# # data = Dict(i[1] => i[2:end] for i in eachcol(data_matrix))


# # Set-up the priors
# αprior = Normal(1, 0.3) # Geometry scaling factor
# βprior = Normal(1, 0.3) # Geometry scaling factor
# γprior = Normal(1, 0.3) # Tp0 scaling factor
# λprior = Normal(1, 0.3) # cp scaling factor

# priors = (α=αprior, β=βprior, γ=γprior, λ=λprior)

# # Create the model
# model = run_onefiber([data["Vlvs"]/ml data["plvs"]/mmHg], priors, constants)

# # Run adaptive Markov chain
# out_adaptive = RAM_sample(
#     model, # log target function
#     [1., 1., 1., 1.],                             # Initial values
#     0.3,                              # Scaling factor #[0.05 0; 0 0.001],
#     10_000                           # Number of runs
# )






