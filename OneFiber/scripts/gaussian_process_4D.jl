using ScikitLearn
using Plots
using CSV, DataFrames, JSON
using Statistics, Distributions, LinearAlgebra
using LazySets, ProgressBars
using LaTeXStrings
using OneFiber


# Import relevant Python submodules
@sk_import linear_model: LogisticRegression
@sk_import gaussian_process: kernels
@sk_import gaussian_process: GaussianProcessRegressor

function read_json(file)
    open(file,"r") do f
        global inDict
        inDict = JSON.parse(f)
    end
    return inDict
 end


# Load the data
# filename(p) = "output/geometry variation/parameters_P$p.csv"
function filename(p)
    if p < 28
        filen = "output/geometry SVD/parameters_P$p.csv"
    elseif p >= 28 && p < 34
        filen = "output/geometry SVD/parameters_patient$(p-27).csv"
    else
        filen = "output/geometry SVD/parameters_patientspecific$(p-33).csv"
    end
    return filen
end

function initVolumes(p)
    if p < 28
        point = string(p)
        filenVcav  = "output/geometry SVD/Cavity_volumes.json"
        filenVcav0 = "output/geometry SVD/GPA_cavity_volumes.json"
        filenVwall = "output/geometry SVD/GPA_wall_volumes.json"
    elseif p >= 28 && p < 34
        point = string(p-27)
        filenVcav  = "output/geometry SVD/Patient_Cavity_volumes.json"
        filenVcav0 = "output/geometry SVD/Patient_GPA_cavity_volumes.json"
        filenVwall = "output/geometry SVD/Patient_GPA_wall_volumes.json"
    else
        point = string(p-33)
        filenVcav  = "output/geometry SVD/Patientspecific_Cavity_volumes.json"
        filenVcav0 = "output/geometry SVD/Patientspecific_GPA_cavity_volumes.json"
        filenVwall = "output/geometry SVD/Patientspecific_GPA_wall_volumes.json"
    end

    Vcavity  = read_json(filenVcav)
    Vcavity0 = read_json(filenVcav0)
    Vwall    = read_json(filenVwall)

    constants = Constants()

    ncycles = 12 # unless patient 10 is considered, then 8
    tstart  = 0.004 # [s]
    trelax  = 0.0   # [s]
    
    (; Vtotal, Vven0, Vart0, Cven, Cart) = constants
    pven = 1600
    part = ( Vtotal - Vcavity[point]*1e3 - Vven0 - Vart0 - Cven*pven ) / Cart
    initial_model = Dict{String, Float64}("Vlv"  => Vcavity[point]*1e3, "lc" => 1.5, "plv" => pven, "part" => part,
                                          "Vlv0" => Vcavity0[point]*1e3, "Vwall" => Vwall[point]*1e3, 
                                          "ncycles" => ncycles, "tstart" => tstart, "trelax" => trelax)

    return initial_model
end

function load_csv(filename) 
    DataF     = CSV.read(filename, DataFrame)
    keynames  = names(DataF)[1:end-1] # Exculde the ncycle values (is int)
    MyData    = Dict( keyname => eval( Meta.parse( Matrix(DataF)[i] ) ) for (i, keyname) in enumerate(keynames) )
    # for (i, key) in enumerate(keys)
    #     MyData[key] = eval( Meta.parse( Matrix(Data)[i] ) );
    # end
    return MyData
end

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end


struct GPdata
    gaussProc
    lengthScales::Array{Float64}
    weight::Float64
end



## Function that constructs the GP and performs a fit + predicts reference point (if desired)
function GaussianProcess__(X, ϴ, dϴnoise; lmin=1e-1, lmax=4e1, optimize=false)

    # Normalize the input (prevents floating point issues)
    weight       = 1. #sqrt( sum( ϴ.^2 )) # required when un-normalizing again
    ϴ_normalized = ϴ / weight
    
    lengthscales        = [2.08050444, 1.03951632, 0.59768942, 0.32444523]
    length_scale_bounds = ( (lengthscales[1]*1e-2, lengthscales[1]),
                            (lengthscales[2]*1e-2, lengthscales[2]),
                            (lengthscales[3]*1e-2, lengthscales[3]),
                            (lengthscales[4]*1e-2, lengthscales[4]))


    # Set the Kernel and corresponding bounds
    if optimize
        alpha  = dϴnoise.^2
        kernel = kernels.RBF(length_scale=lengthscales*0.8, length_scale_bounds=length_scale_bounds)
    else
        alpha         = (dϴnoise / weight).^2 .+ 1e-12 # 1e-8
        length_scales = lengthscales #2*[0.616, 0.9]
        kernel    = kernels.RBF(length_scale=length_scales, length_scale_bounds="fixed")#, length_scale_bounds=(lmin, lmax))
    end

    GaussProc = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10, normalize_y=false, random_state=1)

    # Fit GP
    GaussProc.fit(X, ϴ_normalized)
    println("LogM value: ", GaussProc.log_marginal_likelihood_value_)
    # Retrieve length scale
    Lengthscales = GaussProc.kernel_.length_scale

    # Store relevant data
    DataStruct = GPdata(GaussProc, Lengthscales, weight)

    return DataStruct
end


function GaussianProcess_(X, initial_points, candidate_points; nsamples = 10_000, return_sample_stats = false)

    # mask X points
    Xmasked = X[initial_points, :] # Mask the array

    # Load data
    params    = "beta","phi","lambda","gamma"
    covparams = "beta-beta","beta-phi", "phi-phi", "beta-lambda", "phi-lambda", "lambda-lambda", "beta-gamma", "phi-gamma", "lambda-gamma", "gamma-gamma" 
    ϴparams  = Dict(k => zeros(length(initial_points)) for k in params)
    dϴparams = Dict(k => zeros(length(initial_points)) for k in params)
    ϴcorparams  = Dict(k => zeros(length(initial_points)) for k in covparams)
    dϴcorparams = Dict(k => zeros(length(initial_points)) for k in covparams)

    ϴparam_pred   = Dict(k => zeros(length(candidate_points)) for k in params)
    dϴparams_pred = Dict(k => zeros(length(candidate_points)) for k in params)
    ϴcovparam_pred   = Dict(k => zeros(length(candidate_points)) for k in covparams)
    dϴcovparams_pred = Dict(k => zeros(length(candidate_points)) for k in covparams)
    

    ϴparams_Ref  = Dict()
    dϴparams_Ref = Dict()
    ϴsparams_Ref = Dict()

    nmeans = 2 # Number of sections of the sample chain (used to estimate noise level of the sampler (if 0, we obtain a noise of 0))
    for (i, p) in enumerate( vcat(initial_points) ) #candidate_points
        data  = load_csv(filename(p))

        datm    = [ reshape(data[prm], :, nmeans) for prm in params ]
        DataMat = permutedims( cat( datm[1], datm[2], datm[3], datm[4], dims=3), ( 3, 1, 2 ) )

        # Store parameter data
        for par in params
            ϴparams[par][i]  = mean(data[par])
            dϴparams[par][i] = std(data[par]) #std( mean(reshape(data[par], :, nmeans), dims=1) )#std(data[par]

            # if (p ∈ candidate_points) && (p ∉ ϴparams) 
            #     ϴparams_Ref[par]  = mean(data[par])
            #     dϴparams_Ref[par] = std(data[par]) #std( mean(reshape(data[par], :, nmeans), dims=1) ) #
            #     ϴsparams_Ref[par] = data[par]
            # else
            #     ϴparams[par][i]  = mean(data[par])
            #     dϴparams[par][i] = std(data[par]) #std( mean(reshape(data[par], :, nmeans), dims=1) )#std(data[par])
            # end
        end

        # Store covariance data of the parameters
        triu_indc = triu(trues((4,4))) # indices of upper triangle
        for (c, cpar) in enumerate(covparams)
            ϴcmat   = mean( cat([ cor( DataMat[:,:,k], dims=2 ) for k in range(1, nmeans) ], dims=3), dims=1 )[1]
            dϴcmat  = std( cat([cor( DataMat[:,:,k], dims=2 )[:,:,:] for k in range(1, nmeans)]..., dims=3), dims=3 )[:,:,1]
            ϴcorparams[cpar][i]  =  ϴcmat[triu_indc][c]
            dϴcorparams[cpar][i] = dϴcmat[triu_indc][c]
        
            # if p in initial_points
            #     ϴcmat   = mean( cat([ cor( DataMat[:,:,k], dims=2 ) for k in range(1, nmeans) ], dims=3), dims=1 )[1]
            #     dϴcmat  = std( cat([cor( DataMat[:,:,k], dims=2 )[:,:,:] for k in range(1, nmeans)]..., dims=3), dims=3 )[:,:,1]
            #     ϴcorparams[cpar][i]  =  ϴcmat[triu_indc][c]
            #     dϴcorparams[cpar][i] = dϴcmat[triu_indc][c]
            # end
        end


    end

    
    for (i, p) in enumerate( vcat(initial_points) ) #candidate_points
        data  = load_csv(filename(p))
        for par in params
            ϴparams[par][i]  = mean(data[par])
            dϴparams[par][i] = std(data[par])
            # if p in candidate_points
            #     ϴparams_Ref[par]  = mean(data[par])
            #     dϴparams_Ref[par] = std(data[par])
            #     ϴsparams_Ref[par] = data[par]
            # else
                
            # end
        end
    end
    

    ## Fit and predict the parameters using GP
    GPdict  = Dict()
    for par in params
        GPdict[par] = GaussianProcess__(Xmasked, ϴparams[par], dϴparams[par], optimize=true)

        # Predict candidate points
        for (p, p_pred) in enumerate(candidate_points)
            ppred = (X[p_pred,:]' |> Matrix)
            Ypredicted = GPdict[par].gaussProc.predict( ppred , return_std=true)

            ϴparam_pred[par][p]   = Ypredicted[1][1] * GPdict[par].weight # 
            dϴparams_pred[par][p] = Ypredicted[2][1] * GPdict[par].weight # 
        end
    end

    ## Fit and predict the covariance parameters using GP
    for cpar in covparams
        GPdict[cpar] = GaussianProcess__(Xmasked, ϴcorparams[cpar], dϴcorparams[cpar], optimize=true) # We fit the correlation

        # Predict candidate points
        for (p, p_pred) in enumerate(candidate_points)
            ppred = (X[p_pred,:]' |> Matrix)
            Ypredicted = GPdict[cpar].gaussProc.predict( ppred , return_std=true)

            # We predict the covariance
            cpar1, cpar2 = split(cpar, "-")
            ϴcovparam_pred[cpar][p]   = max.( min.( Ypredicted[1][1], 1 ), -1 ) * (dϴparams_pred[cpar1][p]*dϴparams_pred[cpar2][p]) # Enforce correlation constrain & convert to covariance 
            #dϴcovparams_pred[cpar][p] = Ypredicted[2][1] #* GPdict[cpar].weight # 
        end
    end


    ## Fit the parameters using GP
    # FitParams    = Dict()
    # dFitParams   = Dict()
    # LengthScales = Dict()
    # ReferenceLV  = Dict()

    # Cs = LinRange(0.25,2.25,500)
    # Hs = LinRange(0.0,1.6,500)
    # xx, yy = meshgrid(Cs,Hs)
    # xx_yy  = hcat(vec(xx), vec(yy))
    # nreset = 5
    # GaussianProcesses = Dict()
    # for par in params

    #     kernel    = kernels.RBF(length_scale=[1, 1])
    #     GaussProc = GaussianProcessRegressor(kernel=kernel, alpha=dϴparams[par].^2, n_restarts_optimizer=1, normalize_y=false, random_state=1)

    #     GaussProc.fit(Xmasked, ϴparams[par])
    #     GaussianProcesses[par] = GaussProc

    #     LengthScales[par] = GaussProc.kernel_.length_scale
    #     Yp, Sp = GaussProc.predict(xx_yy, return_std=true)

    #     # Store value for interpolated value of the reference LV
    #     ReferenceLV[par] = GaussProc.predict(reshape([1, 1], 1, 2), return_std=true)



    #     # Predict
    #     for (p, p_pred) in enumerate(candidate_points)
    #         ppred = (X[p_pred,:]' |> Matrix)
    #         Ypredicted = GaussProc.predict( ppred , return_std=true)

    #         ϴparam_pred[par][p]  = Ypredicted[1][1]
    #         dϴparams_pred[par][p] = Ypredicted[2][1]
    #     end
    #     #print(f"{par}: {GP.log_marginal_likelihood_value_}")


    #     FitParams[par]  = reshape( Yp, size(xx)[1], size(yy)[1]) #xx.shape[0], xx.shape[1]))
    #     dFitParams[par] = reshape( Sp, size(xx)[1], size(yy)[1]) #.reshape((xx.shape[0], xx.shape[1]))

    # end

    # # masking region
    # # Convex hull
    # ConvH  = convex_hull( [X[i,:] for i in 1:size(X,1)] )
    # mask   = Matrix{Bool}(undef, length(Cs), length(Hs))
    # mask  .= false
    # for i in range(1, length(Cs))
    #     for j in range(1, length(Hs))
    #         mask[i,j] = Singleton([ xx[i,j], yy[i,j] ]) ⊆ VPolygon(ConvH)
    #     end
    # end



    ## Forward analysis ---------------------------------------
    constants = Constants()

    VED_means  = Array{Float64}(undef, length(candidate_points))
    VES_means  = Array{Float64}(undef, length(candidate_points))
    Pmax_means = Array{Float64}(undef, length(candidate_points))
    EF_means   = Array{Float64}(undef, length(candidate_points))

    VED_std  = Array{Float64}(undef, length(candidate_points))
    VES_std  = Array{Float64}(undef, length(candidate_points))
    Pmax_std = Array{Float64}(undef, length(candidate_points))
    EF_std   = Array{Float64}(undef, length(candidate_points))


    Vsamples = Dict()
    Psamples = Dict()
    PVstats  = Dict()
    for (p, p_pred) in enumerate(candidate_points)

        σjσi = ϴcovparam_pred
        Σ    = [ σjσi["beta-beta"][p]   σjσi["beta-phi"][p]   σjσi["beta-lambda"][p]   σjσi["beta-gamma"][p]  ;
                 σjσi["beta-phi"][p]    σjσi["phi-phi"][p]    σjσi["phi-lambda"][p]    σjσi["phi-gamma"][p]   ;
                 σjσi["beta-lambda"][p] σjσi["phi-lambda"][p] σjσi["lambda-lambda"][p] σjσi["lambda-gamma"][p];
                 σjσi["beta-gamma"][p]  σjσi["phi-gamma"][p]  σjσi["lambda-gamma"][p]  σjσi["gamma-gamma"][p] ]
        

        # Force posite-definite
        (eigenvalues, eigenvectors) = eigen(Σ)
        if minimum(eigenvalues) < 0
            Σnew = Σ + (1e-10 .- minimum(eigenvalues))*Diagonal([1, 1, 1, 1])
            println("Relative error positive def. [%]: ", abs( minimum(eigenvalues)/minimum(diag(Σ)))*100)
        else
            Σnew = Σ
            println("Positive-definite matrix")
        end

        println(Σnew)
        # Σ = Diagonal( [dϴparams_pred[par][p]^2 for par in params] ) # Variance = σ^2
        μ = [ϴparam_pred[par][p] for par in params]

        Pdensity = MvNormal(μ, Σnew) 

        println("Running MC sampling point $p_pred")
        # Initialize
        initial = initVolumes(p_pred)
        Vsamples[p_pred], Psamples[p_pred] = forward_cardiac_MC(Pdensity, constants, initial, nsamples=nsamples, ncycles=Int(initial["ncycles"]))
        #print(Vsamples, Psamples)
        PVstats[p_pred] = Dict( "Pmean" => mean(Psamples[p_pred], dims=1), "Pstd" => std(Psamples[p_pred], dims=1), 
                                     "Vmean" => mean(Vsamples[p_pred], dims=1), "Vstd" => std(Vsamples[p_pred], dims=1) )


        Qmeans, Qstd = qois(Vsamples[p_pred], Psamples[p_pred])
        println("Mean values: ", Qmeans["VED"]*1e3, " ", Qmeans["VES"]*1e3, " ", Qmeans["Pmax"], " ", Qmeans["EF"])
        println("STD values: ", Qstd["VED"]*1e3, " ", Qstd["VES"]*1e3, " ", Qstd["Pmax"], " ", Qstd["EF"])

        VED_means[p]  = Qmeans["VED"]
        VES_means[p]  = Qmeans["VES"]
        Pmax_means[p] = Qmeans["Pmax"]
        EF_means[p]   = Qmeans["EF"]

        VED_std[p]  = Qstd["VED"]
        VES_std[p]  = Qstd["VES"]
        Pmax_std[p] = Qstd["Pmax"]
        EF_std[p]   = Qstd["EF"]
    end
    
    if return_sample_stats
        return PVstats
    else
        return VED_std, Pmax_std, EF_std
    end
end


function forward_cardiac_MC(Pdensity, constants, initial; nsamples=100, ncycles=6)

    (; ms, ml, kPa, mmHg, tcycle, δt) = constants

    constants.Vlv0   = initial["Vlv0"]
    constants.Vwall  = initial["Vwall"]
    constants.tstart = initial["tstart"]
    constants.trelax = initial["trelax"]


    # Set Vwall, Vlv0, part
    #initial   = Dict{String, Float64}("Vlv" => 44*ml, "lc" => 1.5, "plv" => 0*mmHg, "part" => 86.257*mmHg)
    #idx_range = ((ncycles-1)*Int(tcycle/δt) + 1):(ncycles*Int(tcycle/δt))



    Vsamples = Array{Float64}(undef, nsamples, 400)
    Psamples = Array{Float64}(undef, nsamples, 400)
    for n in ProgressBar(1:nsamples)
        β, ϕ, λ, γ = rand(Pdensity, 1)
        # println("β: ", β)
        # println("ϕ: ", ϕ)
        # println("λ: ", λ)
        # println("γ: ", γ)
        constants.β = β
        constants.γ = γ
        constants.λ = λ
        constants.ϕ = ϕ

        # Solve the time dependent problem
        time, Vlv, lc, plv, part = SolveOneFiber(constants, ncycles=ncycles, initial=initial, return_eval=false, print_info=false) #, initial=initial
        
        if any( isinf, Vlv )
           Vsamples[n,:] .= Inf
           Psamples[n,:] .= Inf
        else
            valve_idx = return_valve_points_new(plv, Vlv, cycle=ncycles, return_idx=true)
            idx_range = (Int(valve_idx[1,1]) - Int(tcycle/δt)):(Int(valve_idx[1,1])-1) # Neglect last index

            Vsamples[n,:] = Vlv[idx_range]
            Psamples[n,:] = plv[idx_range]
        end

    end


    ## There exists the posibility that Vsamples and Psamples contain NaNs (when the simulation diverged given the input values)
    # Filter them out but correct the standard deviation and mean accordingly!
    infrows  = any( isinf, Vsamples; dims=2 ) # Should be same for Psamples
    maskinfs = .!vec(infrows)
    # println(nanrows)
    # println(Vsamples)
    # Remove the NaN rows
    println("Found $(sum(vec(infrows)))/$(size(Vsamples)[1]) Inf rows, these are removed.")
    Vsamples = Vsamples[maskinfs,:]
    Psamples = Psamples[maskinfs,:]

    return Vsamples, Psamples
end


function qois(Vsamples, Psamples)



    VEDs  = maximum(Vsamples, dims=2)
    VESs  = minimum(Vsamples, dims=2)
    Pmaxs = maximum(Psamples, dims=2)
    EFs   = (VEDs .- VESs)./VEDs

    # Means
    VED_mean  = mean(VEDs)
    VES_mean  = mean(VESs)
    Pmax_mean = mean(Pmaxs)
    EF_mean   = mean(EFs)
    Means = Dict("VED" => VED_mean, "VES" => VES_mean, "Pmax" => Pmax_mean, "EF" => EF_mean)

    # Std-dev
    VED_std  = std(VEDs) # The std() is by default corrected 1/(n-1)
    VES_std  = std(VESs)
    Pmax_std = std(Pmaxs)
    EF_std   = std(EFs) 
    Stds = Dict("VED" => VED_std, "VES" => VES_std, "Pmax" => Pmax_std, "EF" => EF_std)
    return Means, Stds
end





function point_insertion(X, initial_points, candidate_points, threshold_values; nsamples = 10_000)

    for i in range(1, length(candidate_points))

        VED_std, Pmax_std, EF_std = GaussianProcess_(X, initial_points, candidate_points, nsamples = nsamples)

        # Check which point has highest uncertainty:
        maxVal, maxID = findmax(VED_std)
        println("VED : ", VED_std*1e3)
        println("Pmax: ", Pmax_std)
        println("EF  : ", EF_std)

        println("N-VED : ", VED_std  ./ maximum(VED_std))
        println("N-Pmax: ", Pmax_std ./ maximum(Pmax_std))
        println("N-EF  : ", EF_std   ./ maximum(EF_std))
        if maxVal*1e3 > threshold_values["VED"] # convert to [ml]
            initial_points = vcat(initial_points, candidate_points[maxID]) # Append highest uncertainty point to the initial points
            println("Inserting point ", candidate_points[maxID])
            # Delete the highest uncertainty point from the candidates
            deleteat!(candidate_points, maxID)
        else
            println("Thresholds satisfied")
            break
        end
    end
    return initial_points, candidate_points
end


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

function forward_uncertainty_plot(X, GP_points, predicted_point; nsamples=1_000)

    constants = Constants()
    (; ms, ml, kPa, mmHg, tcycle, δt) = constants

    ## perform forward analysis on the predicted point
    PVstats = GaussianProcess_(X, GP_points, predicted_point, nsamples=nsamples, return_sample_stats=true)
    #predpoint_key = predicted_point[1]

    for predpoint_key in predicted_point
        Ppred_mean = PVstats[predpoint_key]["Pmean"]'/mmHg
        Vpred_mean = PVstats[predpoint_key]["Vmean"]'/ml
        
        Ppred_std = PVstats[predpoint_key]["Pstd"]'/mmHg
        Vpred_std = PVstats[predpoint_key]["Vstd"]'/ml


        ## Perform forward analysis using exact data point data
        prms  = "beta","phi","lambda","gamma"
        data    = load_csv(filename(predpoint_key))
        datm    = hcat( [ reshape(data[prm], 1, :) for prm in prms ] )
        DataMat = cat( datm[1], datm[2], datm[3], datm[4], dims=1)

        # Mean and covariance matrix
        μdata = [mean(data[par]) for par in prms]
        Σdata = cov( DataMat, dims=2)

        Pdensity = MvNormal(μdata, Σdata)
        initial  = initVolumes(predpoint_key)
        Vsamples_data, Psamples_data = forward_cardiac_MC(Pdensity, constants, initial, nsamples=nsamples, ncycles=Int(initial["ncycles"]))

        
        # Stats
        Pdata_mean = mean(Psamples_data, dims=1)'/mmHg
        Vdata_mean = mean(Vsamples_data, dims=1)'/ml
        
        Pdata_std = std(Psamples_data, dims=1)'/mmHg
        Vdata_std = std(Vsamples_data, dims=1)'/ml
        
        
        PVdata_mean = Dict("Vlvs" => Vdata_mean, "plvs" => Pdata_mean)
        PVpred_mean = Dict("Vlvs" => Vpred_mean, "plvs" => Ppred_mean)


        ## Postprocessing
        data_error_bound  = Dict("Vlv95_lower" => Vdata_mean-2*Vdata_std, "Vlv95_upper" => Vdata_mean+2*Vdata_std, "plv95_lower" => Pdata_mean-2*Pdata_std, "plv95_upper" => Pdata_mean+2*Pdata_std)
        pred_error_bound  = Dict("Vlv95_lower" => Vpred_mean-2*Vpred_std, "Vlv95_upper" => Vpred_mean+2*Vpred_std, "plv95_lower" => Ppred_mean-2*Ppred_std, "plv95_upper" => Ppred_mean+2*Ppred_std)

        # Calculate normals
        normal_pred  = compute_normal_PV(Ppred_mean, Vpred_mean)
        normal_data  = compute_normal_PV(Pdata_mean, Vdata_mean)


        #
        PV95_data       =  (normal_data.*[Vdata_std Pdata_std])#(normal.*[σnoise_Vlv σnoise_Plv])
        PV95_data_upper =  [Vdata_mean Pdata_mean] .+ 2*PV95_data
        PV95_data_lower =  [Vdata_mean Pdata_mean] .- 2*PV95_data

        PV95_pred       =  (normal_pred.*[Vpred_std Ppred_std])#(normal.*[σnoise_Vlv σnoise_Plv])
        PV95_pred_upper =  [Vpred_mean Ppred_mean] .+ 2*PV95_pred
        PV95_pred_lower =  [Vpred_mean Ppred_mean] .- 2*PV95_pred

        # Create Tet polygons
        XVerts_data_half1 = [PV95_data_upper[1:(end-1),1] PV95_data_upper[2:end,1] PV95_data_lower[1:(end-1),1]]
        YVerts_data_half1 = [PV95_data_upper[1:(end-1),2] PV95_data_upper[2:end,2] PV95_data_lower[1:(end-1),2]]

        XVerts_data_half2 = [PV95_data_lower[1:(end-1),1] PV95_data_lower[2:end,1] PV95_data_upper[2:(end),1]]
        YVerts_data_half2 = [PV95_data_lower[1:(end-1),2] PV95_data_lower[2:end,2] PV95_data_upper[2:(end),2]]

        XVerts_pred_half1 = [PV95_pred_upper[1:(end-1),1] PV95_pred_upper[2:end,1] PV95_pred_lower[1:(end-1),1]]
        YVerts_pred_half1 = [PV95_pred_upper[1:(end-1),2] PV95_pred_upper[2:end,2] PV95_pred_lower[1:(end-1),2]]

        XVerts_pred_half2 = [PV95_pred_lower[1:(end-1),1] PV95_pred_lower[2:end,1] PV95_pred_upper[2:(end),1]]
        YVerts_pred_half2 = [PV95_pred_lower[1:(end-1),2] PV95_pred_lower[2:end,2] PV95_pred_upper[2:(end),2]]


        data_error_band  = Dict("Xverts1" => XVerts_data_half1, "Yverts1" => YVerts_data_half1, "Xverts2" => XVerts_data_half2, "Yverts2" => YVerts_data_half2)
        pred_error_band  = Dict("Xverts1" => XVerts_pred_half1  , "Yverts1" => YVerts_pred_half1  , "Xverts2" => XVerts_pred_half2  , "Yverts2" => YVerts_pred_half2)

        
        # Save to csv for python post processing
        data_dict = Dict("Pdata_mean" => Pdata_mean, "Pdata_std" => Pdata_std, "Pdata_normal_std" => PV95_data[:,2], 
                        "Vdata_mean" => Vdata_mean, "Vdata_std" => Vdata_std, "Vdata_normal_std" => PV95_data[:,1],
                        "Ppred_mean" => Ppred_mean, "Ppred_std" => Ppred_std, "Ppred_normal_std" => PV95_pred[:,2],
                        "Vpred_mean" => Vpred_mean, "Vpred_std" => Vpred_std, "Vpred_normal_std" => PV95_pred[:,1],)

        #CSV.write(string( "output/PVdata_predictedpoint", predpoint_key, "no_insertion.csv"), data_dict)
        CSV.write(string( "output/PVdata_predictedpoint", predpoint_key, "with_insertion.csv"), data_dict)



        ## Post processing
        post = Postprocessing()
        post.PV_loop(PVdata_mean, PVpred_mean;    data_error_band  = data_error_band , model_error_band  = pred_error_band)

        post.PV_trace(PVdata_mean, PVpred_mean;    data_error_bound = data_error_bound, model_error_bound = pred_error_bound)
    end

    return (Ppred_mean, Vpred_mean, Ppred_std, Vpred_std, PV95_pred), (Pdata_mean, Vdata_mean, Pdata_std, Vdata_std, PV95_data)
end

# X = [0.43 0.96; 0.75 0.45; 1.13 0.00;
#      0.75 1.24; 1.14 0.60; 1.54 0.00;
#      1.09 1.50; 1.53 0.75; 1.95 0.00;
#      1.00 1.00]

X = [-4.43931208e-01 -5.28059218e-02  4.54503436e-02 2.87595523e-02;
    -5.58295672e-01  1.65090145e-01  4.50682011e-02 -7.19714838e-02;
    -7.40994803e-01 -1.94544955e-01 -9.92093632e-02 -8.37902259e-04;
    -3.46761660e-01 -1.80223259e-01 -1.82040744e-02  2.67682897e-02;
     1.62831380e-01  2.56439741e-02  1.02035820e-01  4.30343668e-02;
    -6.75380058e-01  2.51892210e-02 -9.40040667e-03 -2.61291057e-02;
    -7.25072027e-01 -9.37547088e-02 -1.36501143e-01  3.98950595e-02;
     1.33242515e-01  1.16961426e-01  1.61835039e-01  7.91616873e-02;
    -1.62214825e-01 -2.37840868e-01  8.07188189e-02 -2.63211545e-02;
    -8.77420839e-01 -1.08533651e-01 -1.37009672e-01  1.07349000e-02;
    -6.04255428e-01 -2.27833307e-01 -9.42910793e-02  1.11107124e-02;
    -4.81049813e-01  1.63355534e-01  7.32866700e-02 -6.50383639e-02;
    -5.76549352e-01  4.61147915e-02  1.99362389e-02  2.03488045e-03;
    -5.50149892e-01  8.24046191e-02 -1.95348556e-02 -6.86507660e-02;
    -2.94310721e-01  9.38649271e-02  1.10889301e-01 -9.67626081e-03;
    -3.54503764e-01  6.39278651e-02  2.46866021e-02  8.75672671e-02;
    -6.59182146e-01  9.85507775e-02 -4.29811888e-02 -1.26543375e-02;
     1.04690986e-01 -2.11794432e-01  1.53885167e-01 -5.10697180e-02;
    -2.24744511e-01 -8.44587505e-02  7.46522532e-02  3.50025641e-02;
    -1.01543319e-01 -3.22600866e-01  6.97671989e-02 -6.24694408e-02;
    -2.62509685e-01  2.42025544e-02  1.13062278e-01  2.96796951e-02;
    -1.03603633e-01  1.31241356e-01  3.33731313e-02  3.76163366e-03;
    -6.93779644e-01  1.04712230e-02 -7.55194364e-02 -4.02739865e-02;
    -2.97294930e-01 -2.12583180e-01  3.60222165e-03  1.41412621e-02;
    -6.67711406e-01  1.21425449e-01 -7.52512646e-03 -7.46553455e-02;
     9.86055178e-02  1.97157292e-01  1.52717992e-01  3.50578087e-02;
    -2.19464811e-01 -2.05199930e-01  4.37145311e-03 -7.00881764e-03;
    -0.40945329      0.05624277      0.05453193      0.00463231;
    -0.39472902      0.05756561      0.03706031     -0.01106285;
    -0.457258       -0.00194585      0.03239358     -0.01060212;
    -0.17014093     -0.02769511      0.08370594      0.02439406;
    -0.49874722      0.05426793     -0.00448095     -0.02662656;
    -0.44747397      0.01821244      0.05167868      0.02874061;
    -0.20820573632376646 -0.05716343983208385 0.05205683680729817 0.04799775293788662]



# initial_points   = [1, 3, 7, 9]#6, 7, 8, 9]
# candidate_points = [1, 2, 4, 5, 6, 8, 10]#[10]
# threshold_values = Dict( "VED"  => 1.0, # [ml]
#                          "VES"  => 1.0, # [ml] 
#                          "EF"   => 1.0, # [%]
#                          "Pmax" => 2.0 )# [mmHg]
# initial_points_end, candidate_points_end = point_insertion(X, initial_points, candidate_points, threshold_values; nsamples = 100)


initial_points   = [i for i in range(1,34)]#6, 7, 8, 9]
candidate_points = [33]#[28, 33] #[34] ##[10]
filter!(f -> f ∉ candidate_points, initial_points)
candidate_points = [28, 33]
PVpred, PVdata = forward_uncertainty_plot(X, initial_points, candidate_points; nsamples = 10_000)# 5000)







## Postprocessing
function plot_contour(x, y, data, param, mask)
    dataM = copy(data[param])
    dataM[.!mask] .= NaN
    surf = contourf(x, y, dataM', xlabel=L"\tilde{C}", ylabel=L"\tilde{H}", title=param)
    return surf
end
# surf = Dict()
# for par in params
#     surf[par] = plot_contour(Cs, Hs, FitParams, par, mask)
# end
# figure = plot(surf["beta"], surf["phi"], surf["gamma"], surf["lambda"], layout=(2,2), size=(2*250,2*250), display_type=:gui)
