using LinearAlgebra


"""
    Newton(K, res, x0, xinit, cons=[], tol=1e-8) -> xsol, residual

Solves the system of equations, K⋅Δx=res, in an iterative manner until the 
residual, |res|<tol, reaches a specific tolerance value....

x0    = Previous time-step values
xinit = Initial values used as starting point for Newton iterations



niter_max = Maximum number of allowed newton iterations before returning infinity ∞

"""
function Newton(K, res, x0, xinit; cons=Dict(), tol=1e-8, return_residual=false, print_info=true, niter_max=500)

    # Set the masking info (if any)
    x        = xinit
    mask_key = Dict{String, Int}("Vlv" => 1, "lc" => 2, "plv" => 3, "part" => 4)
    imask    = Vector{Int64}()
    
    inomask  = [1, 2, 3, 4]
    mask     = [false false false false]
    for (key, value) in cons
        push!(imask, mask_key[key])
        setfield!(x, Symbol(key), value) # Make sure the constraints are assigned to the x struct as well
    end
    filter!(e->e ∉ imask, inomask) # Inplace change of the no mask indices vector

    # mask[imask] .= true
    # mask = .!mask
    # nmask = length(imask) # Number of masks
    # nsolv = 4 - nmask # number of variables to solve for
    # Mmask = (mask.*mask') # matrix mask

    # Initialize
    global i = 0
    global resnorm = 1e3
    dsol = zeros(4) # Solution increments
    residual_norms = Vector{Float64}()
    
    # Evaluate and mask the stiffness matrix and residual vector
    # if nmask != 0
    #     global Kmask(x) = reshape( K(x, x0)[Mmask], (nsolv, nsolv) )
    #     global rmask(x) = res(x, x0)[mask']
    # else
    #     global Kmask(x)   = K(x, x0)
    #     global rmask(x)   = res(x, x0)
    # end

    # Kmask(x) = reshape( K(x, x0)[Mmask], (nsolv, nsolv) )
    # rmask(x) = res(x, x0)[mask']

    # Start Newton loop
    while resnorm > tol

        # Evaluate and mask the stiffness matrix and residual vector
        #println(x)
        Kmask   = K(x, x0)
        rmask   = res(x, x0)

        # Check for nan values
        if any( isnan.(Kmask) ) || any( isnan.(Kmask) )
            global resnorm = Inf
            println("Kmask: ", Kmask)
            println("rmask: ", rmask)
            break
        end
        
        # if nmask != 0
        #     global Kmask = reshape( K(x, x0)[Mmask], (nsolv, nsolv) )
        #     global rmask = res(x, x0)[mask']
        # else
        #     global Kmask   = K(x, x0)
        #     global rmask   = res(x, x0)
        # end


        # Solve the system of equations
        Δsol = - Kmask \ rmask 
        
        #Δsol = - Kmask(x) \ rmask(x) 

        # Set type same as solution type (important when doing automatic differentiation)
        # if typeof(Δsol[1]) != typeof(dsol[1])
        #     dsol = zeros(typeof(Δsol[1]), 4) 
        # end

        # Update struct values
        @inbounds dsol[inomask] = Δsol[inomask] # Retrieve the 4 x 1 shape
        
        
        x.Vlv  += dsol[1]
        x.lc   += dsol[2]
        x.plv  += dsol[3]
        x.part += dsol[4]

        
        ## TODO Set individual norm values
        # global resnorm = norm(rmask[inomask])
        # global i += 1
        if i > (niter_max-1) # Obtained max allowed Newton iterations, return ∞
            global resnorm = Inf
            break
        else
            ## TODO Set individual norm values
            @inbounds global resnorm = norm(rmask[inomask])
            global i += 1
        end

        if print_info
            println("$(i) Newton iteration, residual norm: $(resnorm)")
        end

        if return_residual
            push!(residual_norms, resnorm)
        end


    end

    
        
    if return_residual
        return x, resnorm
    else
        return x
    end

end


function SolveOneFiber(constants; ArrType=Float64, initial=Dict{String, Float64}("Vlv" => 0.044, "lc" => 1.5, "plv" => 0., "part" => 11_499.9875), ncycles=1, return_eval=true, print_info=true)
    #println(ArrType)

    # Initialize
    ∂ℛ, ℛ, Func = getResidual_matrix_vector(constants, return_eval=true)
    (; ms, ml, kPa, mmHg, δt, tstart, tcycle, trelax) = constants

    # Initial values
    Vlv0   = initial["Vlv"]
    lc0    = initial["lc"]
    plv0   = initial["plv"]
    part0  = initial["part"]

    # Set time array
    tmax   = tcycle*ncycles
    time   = 0:δt:tmax

    # Empty solution arrays (to store the results)
    Vlv  = zeros(ArrType, length(time))
    lc   = zeros(ArrType, length(time))
    plv  = zeros(ArrType, length(time))
    part = zeros(ArrType, length(time))
    Vlv[1]=Vlv0;lc[1]=lc0;plv[1]=plv0;part[1]=part0;

    ## Initialize init values
    x0    = Quantities(Vlv=Vlv0, lc=lc0, plv=plv0, part=part0, ta=time[1]) # Struct: Quantity values at t=0 [s]
    cons  = Dict{String, ArrType}("lc" => Func.ls(Vlv0)) #Dict{String, Float64}("lc" => Func.ls(Vlv0)) # Constraints dictionary
    xinit = deepcopy(x0) # Struct
    sol   = deepcopy(x0) # Struct
    ncycle_old = 0
    for (i, t) ∈ zip(Iterators.countfrom(2), time[2:end])

        # Set cycle time (activation time ta)
        ta, ncycle_old  = active_time(t, tcycle, ncycle_old, trelax)
        xinit.ta = ta - tstart

        # Set constraints
        if ta < trelax
            cons["lc"] = Func.ls(sol.Vlv)
        else
            delete!(cons, "lc") # Remove contraints
        end


        # Solve the system of equations
        sol, resnorm = Newton(∂ℛ, ℛ, x0, xinit, cons=cons, print_info=print_info, return_residual=true)

        # Store the result in arrays
        @inbounds Vlv[i]  = sol.Vlv
        @inbounds lc[i]   = sol.lc
        @inbounds plv[i]  = sol.plv
        @inbounds part[i] = sol.part

        # Set solved values at t equal to t-1 for next iteration
        x0.Vlv  = sol.Vlv
        x0.lc   = sol.lc
        x0.plv  = sol.plv
        x0.part = sol.part
        x0.ta   = ta

        # Set initial Newton iteration value (equal to solution of previous iter)
        xinit.Vlv  = sol.Vlv
        xinit.lc   = sol.lc
        xinit.plv  = sol.plv
        xinit.part = sol.part

        if resnorm == Inf
            Vlv[:]  .= Inf
            lc[:]   .= Inf
            plv[:]  .= Inf
            part[:] .= Inf
            break
        end
    end

    if return_eval
        return (time, Vlv, lc, plv, part), Func
    else
        return time, Vlv, lc, plv, part
    end
end


function active_time(t, tcycle, ncycle_old, trelax)
    ## Activate electric if necessary (required when simulating multiple loops) 
    ncycle_new =  floor( t / tcycle )
    if ncycle_new > ncycle_old # we entered a new cycle

        if (t - ncycle_old*tcycle) < (tcycle + trelax)
            ta = t - ncycle_old*tcycle
        else
            ncycle_old = ncycle_new
            ta = t - ncycle_new*tcycle
        end

    else
        ta = t - ncycle_new*tcycle
    end
    return ta, ncycle_old
end
