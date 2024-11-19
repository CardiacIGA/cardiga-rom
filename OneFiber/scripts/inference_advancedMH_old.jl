using AdvancedMH, MCMCChains
using OneFiber, Plots, StatsPlots, LaTeXStrings, AxisArrays
using DelimitedFiles, Distributions, LinearAlgebra

# Include adaptive function
include("adaptive_inference.jl")
include("Postprocessing.jl")

# mutable struct AcceptanceInfo
#     lastlogposterior::Float64
#     acceptedcounter::Integer
#     rejectedcounter::Integer
#     AcceptanceInfo() = new(0.,0,0)
# end

# acceptance_info = AcceptanceInfo()

# function sampler_info(rng, model, sampler, sample, state, iteration)

#     if (sample.lp == acceptance_info.lastlogposterior)
#         acceptance_info.rejectedcounter+=1
#     else
#         acceptance_info.acceptedcounter+=1
#     end

#     acceptance_info.lastlogposterior = sample.lp
# end




# mutable struct AcceptanceInfo
#     lastlogposterior::Float64
#     acceptedcounter::Integer
#     rejectedcounter::Integer
#     rejectedsamplesIdx::Array{Integer}
#     #AcceptanceInfo() = new(0.,0,0,[0])
# end

# AcceptanceInfo(nsamples::Int) = AcceptanceInfo(0.,0,0,collect(range(1,nsamples)))

# function sampler_info(nsamples)

#     # Set the number of samples
#     acceptance_info = AcceptanceInfo(nsamples)

#     let acceptance_info=acceptance_info
#         function sampler_info(rng, model, sampler, sample, state, iteration)
#             if (sample.lp == acceptance_info.lastlogposterior)
#                 acceptance_info.rejectedcounter+=1
#                 acceptance_info[acceptance_info.rejectedcounter + acceptance_info.acceptedcounter] = -1 # -1 indicates whether this sampled index was rejected
#             else
#                 acceptance_info.acceptedcounter+=1
#             end

#             acceptance_info.lastlogposterior = sample.lp
#         end
#     end
# end

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
point       = "5" 
dimension   = Dict("1" => "L11_H00", "2" => "L145_H00", "3" => "L185_H00", "4" => "L085_H035", "5" => "L115_H05",
                    "6" => "L15_H065", "7" => "L06_H085", "8" => "L085_H105", "9" => "L12_H12", "10" => "L10_H10") 
filenameIGA = string("data/geometry variation/results_IGA_GtypeP",point,"_", dimension[point],".csv") #
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
σnoise_Plv = 0.5*(120 -  0)*σ_Plv # Noise in [mmHg]

# Set noise decrease/increase instances
valve_points = return_valve_points(data_input[:,2], data_input[:,1], return_idx=true)
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
σfrac  = 0.6
σnoise_Vlv_arr = ( σfrac*σnoise_Vlv ) .* fnoise.(range(1,400), k=σslope, t1=t1, t2=t2, t3=t3) .+ (1-σfrac)*σnoise_Vlv
Σnoise_Vlv_mat = σnoise_Vlv_arr .* σnoise_Vlv_arr'

l     = 1 # length scale
ts_05 = 399 #idx_range[end-1]
ts    = abs.(collect(range( -ts_05, ts_05 )))
Σtime = stack([ts[(ts_05+2-i):(end+1-i)] for i in range(1,size(ts[ts_05:end],1)-1)],dims=2)
Σplv  = σnoise_Plv^2 * exp.( -δt/( tcycle - δt ) .* l * Σtime )
ΣVlv  = Σnoise_Vlv_mat .* exp.( -δt/( tcycle - δt ) .* l * Σtime )

## Run the adaptive metropolis
out_adaptive = run_RobustAdaptiveMetropolis(data_input, priors, constants; ncycles=ncycles, 
                                      target_accept=target_accept, nsamples=nsamples_adaptMH, restart_every_nth=resets_everyn_adaptMH, 
                                      ΣVlv=ΣVlv, Σplv=Σplv)
# println(out_adaptive.M)
# out_adaptive2 = run_AdaptiveMetropolis(data_input, priors, constants; ncycles=ncycles, 
#                                        target_accept=target_accept, nsamples=nsamples_adaptMH, restart_every_nth=resets_everyn_adaptMH, 
#                                        ΣVlv=ΣVlv, Σplv=Σplv)


# constants.β = 0.28
# constants.γ = 2.20
# constants.λ = 1.64
# constants.ϕ = 1.87 

# time, Vlvmean, lc, plvmean, part = SolveOneFiber(constants, ncycles=ncycles, return_eval=false, print_info=false) #, initial=initial




# Run the MH algorithm
# Σcov =  [1.37883e-5  2.11924e-6  -4.25427e-7  -8.77065e-6;
#          2.11924e-6  2.87427e-6   6.30631e-7   4.35441e-7;
#         -4.25427e-7  6.30631e-7   0.00082905  -0.00017462;
#         -8.77065e-6  4.35441e-7  -0.00017462   4.40288e-5]

# Σcov = [ 0.00110744    0.000130034  -0.000710198  -0.000602194;
#          0.000130034   0.000220065  -5.47841e-5    8.21326e-5;
#         -0.000710198  -5.47841e-5    0.025466     -0.00488034;
#         -0.000602194   8.21326e-5   -0.00488034    0.00157013]

# 6 Cycles
# Σcov = [ 0.000731008   0.000118826  -0.000581549  -0.000322165;
#          0.000118826   0.000146987   5.66638e-5   -1.00034e-5;
#         -0.000581549   5.66638e-5    0.00428894   -0.0014629;
#         -0.000322165  -1.00034e-5   -0.0014629     0.00102044]
# Σcov = [ 0.00343867    0.00139425   0.000569613  -0.00454299;
#          0.00139425    0.0406343   -0.00388161   -0.00150029;
#          0.000569613  -0.00388161   0.000586097  -0.000655454;
#         -0.00454299   -0.00150029  -0.000655454   0.0062861]

## Reference geometry
# Σcov = [0.000754954   0.000717966   7.60715e-5   -0.000504544;
#         0.000717966   0.0427102    -0.00403369   -0.000349375;
#         7.60715e-5   -0.00403369    0.000431238  -4.18853e-5;
#        -0.000504544  -0.000349375  -4.18853e-5    0.000361235]

# Point 1 6th cycle, init=[0.282275,  2.20489,  1.64247,  1.86797]
# Σcov = [ 0.00137957   -0.000319061   0.000438728  -0.000968395;
#         -0.000319061   0.0103388    -0.00302756    0.000293528;
#          0.000438728  -0.00302756    0.00106812   -0.000268267;
#         -0.000968395   0.000293528  -0.000268267   0.000722659]

# Point 2 6th cycle, init=[0.35825  2.09248  1.51079  1.65632]
# Σcov = [ 0.00161756   3.30917e-5   0.00042741   -0.00108862;
#          3.30917e-5   0.0105062   -0.00292329    5.46712e-5;
#          0.00042741  -0.00292329   0.00100747   -0.000258565;
#         -0.00108862   5.46712e-5  -0.000258565   0.000770534]

# Point 3 6th cycle, init=[0.43387,  1.99421,  1.42266,  1.48455]
#  Σcov = [ 0.000869455   4.71168e-5   0.000196867  -0.00059746;
#           4.71168e-5    0.0103037   -0.0029112    -1.69583e-5;
#           0.000196867  -0.0029112    0.000942194  -9.34199e-5;
#          -0.00059746   -1.69583e-5  -9.34199e-5    0.000445143]

# Point 4 6th cycle, init=[0.240028  2.1973  1.65784  1.91686]
# Σcov = [ 0.00122678   -0.000150648   0.000328711  -0.000872355;
#         -0.000150648   0.0111222    -0.00328605    0.000169064;
#          0.000328711  -0.00328605    0.00112821   -0.00019234;
#         -0.000872355   0.000169064  -0.00019234    0.000663806]

# Point 5 6th cycle, init=[0.334939  2.07793  1.50045  1.66242]
# Σcov = [0.00151892    6.58494e-5   0.000401609  -0.00101552;
    #     6.58494e-5    0.0113869   -0.00313079    4.95499e-5;
    #     0.000401609  -0.00313079   0.00105956   -0.00024035;
    #    -0.00101552    4.95499e-5  -0.00024035    0.000718745]

# Point 6 6th cycle, init=[0.437043  1.97572  1.40601  1.45938]
# Σcov = [ 0.00115001    0.000136925   0.000261296  -0.000760203;
#          0.000136925   0.0107854    -0.00296283   -3.40345e-5;
#          0.000261296  -0.00296283    0.000959186  -0.000144405;
#         -0.000760203  -3.40345e-5   -0.000144405   0.000534961]

# Point 7 6th cycle, init=[0.224971  2.17017  1.63505  1.90454]
# Σcov = [ 0.00173284   -6.40593e-5   0.000437383  -0.00122291;
#         -6.40593e-5    0.0107772   -0.00319199    8.48328e-5;
#          0.000437383  -0.00319199   0.00114925   -0.000256063;
#         -0.00122291    8.48328e-5  -0.000256063   0.000907992]

# Point 8 6th cycle, init=[0.3268  2.05371  1.47642  1.6378]
# Σcov = [ 0.000974468   4.96514e-5   0.00022043   -0.000670816;
#          4.96514e-5    0.0101076   -0.00277052    6.78078e-5; 
#          0.00022043   -0.00277052   0.000884274  -0.000134229;
#         -0.000670816   6.78078e-5  -0.000134229   0.000496764]

# Point 9 6th cycle, init=[0.436667  1.96141  1.3871  1.43621]
# Σcov = [ 0.00127483   -2.46956e-5   0.000358502  -0.000828462;
#         -2.46956e-5    0.010505    -0.00292096    6.56343e-5;
#          0.000358502  -0.00292096   0.000981164  -0.000199698;
#         -0.000828462   6.56343e-5  -0.000199698   0.000572807]

# Point 10 6th cycle, init=[0.372687  2.02716  1.44912  1.56704]
# Σcov = [ 0.000881657  -1.36751e-6   0.000234966  -0.000594135;
#         -1.36751e-6    0.0108725   -0.00298304    9.08383e-5;
#          0.000234966  -0.00298304   0.000953914  -0.000134865;
#         -0.000594135   9.08383e-5  -0.000134865   0.00043643]


#σ0         = [std(prior)^2 for prior in priors] 
# Σcov       = out_adaptive.M #Diagonal(σ0) #out_adaptive.M
Σcov = [0.00151892    6.58494e-5   0.000401609  -0.00101552;
        6.58494e-5    0.0113869   -0.00313079    4.95499e-5;
        0.000401609  -0.00313079   0.00105956   -0.00024035;
       -0.00101552    4.95499e-5  -0.00024035    0.000718745]

samples    = Samples(true,                                                   # Set true if samples should be stores
                     Array{Float64}(undef, nsamples_MH, size(data["Vlvs"],1)),  # Initialize stored Vlv array
                     Array{Float64}(undef, nsamples_MH, size(data["plvs"],1)),  # Initialize stored plv array
                     1,                                                      # Starting index 
                     nburnin_MH)                                             # Number of burn-in samples
LogP_model = return_LogP_Onefiber(data_input, priors, constants; ncycles=ncycles, ΣVlv=ΣVlv, Σplv=Σplv, samples=samples)
model      = DensityModel( LogP_model )
initial    = vec([0.334939  2.07793  1.50045  1.66242]) #vec(mean(out_adaptive.chain, dims=1)) # Ref: [0.817, 4.496, 0.879, 1.084]
sampler    = RWMH(MvNormal(zeros(size(Σcov,1)), Σcov))
chain      = sample(model, sampler, nsamples_MH, discard_initial=nburnin_MH, param_names = ["β", "γ", "λ", "ϕ"], initial_params = initial, chain_type = Chains)


# Filter chainbased on acceptance
# range_idx = collect(range(1,nsamples_MH))
# lps = Array(chain[:lp])
# for (i, lpi) in enumerate(lps)
#     if i > 1
#         if lpi == lps[i-1]
#             range_idx[i] = -1 
#         end
#     end
# end
# # Ensure only the accepted samples are stored(!)
# accepted_idx = range_idx.>0
# samples.Vlv = samples.Vlv[accepted_idx,:]
# samples.Plv = samples.Plv[accepted_idx,:]


## Post-processing
#αs  = Array(getproperty(get(chain, Symbol("α")), Symbol("α")))
βs  = Array(getproperty(get(chain, Symbol("β")), Symbol("β")))
γs  = Array(getproperty(get(chain, Symbol("γ")), Symbol("γ")))
λs  = Array(getproperty(get(chain, Symbol("λ")), Symbol("λ")))
ϕs  = Array(getproperty(get(chain, Symbol("ϕ")), Symbol("ϕ")))
#ωs  = Array(getproperty(get(chain, Symbol("ω")), Symbol("ω")))

#lps = get(chain, :lp).lp
# chain_samples = (α=αs, β=βs, γ=γs, λ=λs, ϕ=ϕs)
chain_samples = (β=βs, γ=γs, λ=λs, ϕ=ϕs)

# αarray  = collect(range(start=minimum(αs), stop=maximum(αs), length=100))
βarray  = collect(range(start=minimum(βs), stop=maximum(βs), length=100))
γarray  = collect(range(start=minimum(γs), stop=maximum(γs), length=100))
λarray  = collect(range(start=minimum(λs), stop=maximum(λs), length=100))
ϕarray  = collect(range(start=minimum(ϕs), stop=maximum(ϕs), length=100))
#ωarray  = collect(range(start=minimum(ωs), stop=maximum(ωs), length=100))


# αlogprior   = logpdf.(αprior, αarray)
βlogprior   = logpdf.(βprior, βarray)
γlogprior   = logpdf.(γprior, γarray)
λlogprior   = logpdf.(λprior, λarray)
ϕlogprior   = logpdf.(ϕprior, ϕarray)
#ωlogprior   = logpdf.(ωprior, ωarray)

# Run model with average values ← Final value
# constants.α = mean(αs)
constants.β = mean(βs)
constants.γ = mean(γs)
constants.λ = mean(λs)
constants.ϕ = mean(ϕs)
#constants.ω = mean(ωs)
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

# Extract 95% confidenc einterval for PV-loop
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
post.PV_loop(data_points, model_points;     data_error_band  = data_error_band , model_error_band  = model_error_band)

post.PV_trace(data_points, model_points;    data_error_bound = data_error_bound, model_error_bound = model_error_bound)

post.Chain( chain_samples, priors)


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

                 # Cycle number which was inferred
                 ncycles = ncycles)


# post.write_output(chain_samples; filename=string("output/geometry variation/parameters_P", point, ".csv"))

# Fix postprocess saving
# fileFix = string("output/geometry variation/parameters_P", point, ".csv")
# dataFix = readdlm(fileFix, ',')
# dataF   = Dict(i[1] => i[2:end] for i in eachcol(dataFix))

# betaF   = parse.(Float64,  split( chop( dataF["beta"][1] ; head=1, tail=1), ";")[1:end-2] )
# gammaF  = parse.(Float64,  split( chop( dataF["gamma"][1] ; head=1, tail=1), ";")[1:end-2] )
# lambdaF = parse.(Float64,  split( chop( dataF["lambda"][1] ; head=1, tail=1), ";")[1:end-2] )
# phiF    = parse.(Float64,  split( chop( dataF["phi"][1] ; head=1, tail=1), ";")[1:end-2] )

# Vlv_meanF    = parse.(Float64,  split( chop( dataF["Vlv_mean"][1] ; head=1, tail=1), " "))
# Vlv_95upperF = parse.(Float64,  split( chop( dataF["Vlv_95upper"][1] ; head=1, tail=1), ","))
# Vlv_95lowerF = parse.(Float64,  split( chop( dataF["Vlv_95lower"][1] ; head=1, tail=1), ","))
# Plv_meanF    = parse.(Float64,  split( chop( dataF["Plv_mean"][1] ; head=1, tail=1), " "))
# Plv_95upperF = parse.(Float64,  split( chop( dataF["Plv_95upper"][1] ; head=1, tail=1), ","))
# Plv_95lowerF = parse.(Float64,  split( chop( dataF["Plv_95lower"][1] ; head=1, tail=1), ","))

# function convert_str_2_mat(str) 
#     row_split =  split( chop( str ; head=1, tail=1), ";")
#     matrix = Array{Float64}(undef, size(row_split,1), 2)
#     for (i, str_row) in enumerate(row_split)
#         if str_row[1] == ' '
#             matrix[i,:] = parse.(Float64, split(str_row[2:end], " ") ) 
#         else
#             matrix[i,:] = parse.(Float64, split(str_row, " ") )
#         end
#     end
#     return matrix
# end

# PV_meanF    = convert_str_2_mat(dataF["PV_mean"][1])
# PV_95upperF = convert_str_2_mat(dataF["PV_95upper"][1])
# PV_95lowerF = convert_str_2_mat(dataF["PV_95lower"][1])


# chain_samples = (# Parameters
#                  beta=betaF, gamma=gammaF, lambda=lambdaF, phi=phiF, 

#                  # Pressure-Volume traces
#                  Vlv_mean    = Vlv_meanF,
#                  Vlv_95upper = Vlv_95upperF,
#                  Vlv_95lower = Vlv_95lowerF,
#                  Plv_mean    = Plv_meanF,
#                  Plv_95upper = Plv_95upperF,
#                  Plv_95lower = Plv_95lowerF,

#                  # Pressure-Volume loop
#                  PV_mean    = PV_meanF,     
#                  PV_95upper = PV_95upperF,
#                  PV_95lower = PV_95lowerF,

#                  # Cycle number which was inferred
#                  ncycles = 6)


# beta=βs, gamma=γs, lambda=λs, phi=ϕs, 

# # Pressure-Volume traces
# Vlv_mean    = mean(Vlv_samples, dims=1),
# Vlv_95upper = Vlv95_upper,
# Vlv_95lower = Vlv95_lower,
# Plv_mean    = mean(Plv_samples, dims=1),
# Plv_95upper = plv95_upper,
# Plv_95lower = plv95_lower,

# # Pressure-Volume loop
# PV_mean    = PV95_sim,     
# PV_95upper = PV95_sim_upper,
# PV_95lower = PV95_sim_lower
