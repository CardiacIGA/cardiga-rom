using ScikitLearn
using Plots
using CSV, DataFrames
using Statistics, Distributions, LinearAlgebra
using LazySets, ProgressBars
using LaTeXStrings
using OneFiber


# Import relevant Python submodules
@sk_import linear_model: LogisticRegression
@sk_import gaussian_process: kernels
@sk_import gaussian_process: GaussianProcessRegressor

# Load the data
filename(p) = "output/geometry variation/parameters_P$p.csv"
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
    
    # Set the Kernel and corresponding bounds
    if optimize
        alpha  = 1e-12
        kernel = kernels.RBF(length_scale=[1, 1], length_scale_bounds=(lmin, lmax))
    else
        alpha         = (dϴnoise / weight).^2 .+ 1e-12 # 1e-8
        length_scales = [1.52, 1.5] #2*[0.616, 0.9]
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
        GPdict[par] = GaussianProcess__(Xmasked, ϴparams[par], dϴparams[par])

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
        Vsamples[p_pred], Psamples[p_pred] = forward_cardiac_MC(Pdensity, constants, nsamples=nsamples, ncycles=6)
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


function forward_cardiac_MC(Pdensity, constants; nsamples=100, ncycles=6)

    (; ms, ml, kPa, mmHg, tcycle, δt) = constants

    initial   = Dict{String, Float64}("Vlv" => 44*ml, "lc" => 1.5, "plv" => 0*mmHg, "part" => 86.257*mmHg)
    idx_range = ((ncycles-1)*Int(tcycle/δt) + 1):(ncycles*Int(tcycle/δt))

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
        Vsamples[n,:] = Vlv[idx_range]
        Psamples[n,:] = plv[idx_range]
    end
    return Vsamples, Psamples
end


function qois(Vsamples, Psamples)

    ## There exists the posibility that Vsamples and Psamples contain NaNs (when the simulation diverged given the input values)
    # Filter them out but correct the standard deviation and mean accordingly!


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
    predpoint_key = predicted_point[1]
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
    Vsamples_data, Psamples_data = forward_cardiac_MC(Pdensity, constants, nsamples=nsamples, ncycles=6)

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

    CSV.write("output/PVdata_g5_GP2D_no_g5.csv", data_dict)


    ## Post processing
    post = Postprocessing()
    post.PV_loop(PVdata_mean, PVpred_mean;    data_error_band  = data_error_band , model_error_band  = pred_error_band)

    post.PV_trace(PVdata_mean, PVpred_mean;    data_error_bound = data_error_bound, model_error_bound = pred_error_bound)


    return (Ppred_mean, Vpred_mean, Ppred_std, Vpred_std, PV95_pred), (Pdata_mean, Vdata_mean, Pdata_std, Vdata_std, PV95_data)
end

X = [0.43 0.96; 0.75 0.45; 1.13 0.00;
     0.75 1.24; 1.14 0.60; 1.54 0.00;
     1.09 1.50; 1.53 0.75; 1.95 0.00;
     1.00 1.00]
# initial_points   = [1, 3, 7, 9]#6, 7, 8, 9]
# candidate_points = [1, 2, 4, 5, 6, 8, 10]#[10]
# threshold_values = Dict( "VED"  => 1.0, # [ml]
#                          "VES"  => 1.0, # [ml] 
#                          "EF"   => 1.0, # [%]
#                          "Pmax" => 2.0 )# [mmHg]
# initial_points_end, candidate_points_end = point_insertion(X, initial_points, candidate_points, threshold_values; nsamples = 100)


initial_points   = [1, 2, 3, 4, 6, 7, 8, 9]#6, 7, 8, 9]
candidate_points = [5]#[10]
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
