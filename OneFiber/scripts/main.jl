using DelimitedFiles, CSV, DataFrames, LaTeXStrings, Plots
using OneFiber

function main()
    constants = Constants()
    (; ms, ml, kPa, mmHg) = constants
    # constants.Tp0 = 0.6*kPa
    # constants.ls0 = 1.81

    # #constants.α = 0.9 # Geometric parameter
    # #constants.β = 1.3 # Geometric parameter
    # #constants.γ = 1.4 # Mechanics parameter
    # #constants.λ = 1.2 # Mechanics parameter

    # initial   = Dict{String, Float64}("Vlv" => 44*ml, "lc" => 1.5, "plv" => 0*mmHg, "part" => 86.257*mmHg)

    # # Read data
    # datafr_OF = CSV.File("data/result_OF.csv")|>DataFrame
    # data_OF   = Dict(pairs(eachcol(datafr_OF)))

    # datafr_IGA = CSV.File("data/result_IGA.csv")|>DataFrame
    # data_IGA   = Dict(pairs(eachcol(datafr_IGA)))

    # # Solve the time dependent problem
    # (time, Vlv, lc, plv, part), Func = SolveOneFiber(constants, initial=initial, ncycles=1)

    initial   = Dict{String, Float64}("Vlv" => 44*ml, "lc" => 1.5, "plv" => 0*mmHg, "part" => 86.257*mmHg)
    (time, Vlv_ref, lc, plv_ref, part), Func = SolveOneFiber(constants, initial=initial, ncycles=1)

    initial   = Dict{String, Float64}("Vlv" => 43*ml, "lc" => 1.5, "plv" => 0*mmHg, "part" => 86.257*mmHg)
    constants.Vlv0 = 43*ml
    (time, Vlv, lc, plv, part), Func = SolveOneFiber(constants, initial=initial, ncycles=1)

    display(plot(Vlv_ref/ml, plv_ref/mmHg, linewidth=3, label="Julia ref"))
    plot!(Vlv/ml, plv/mmHg, linewidth=3, label="Julia")
    # Save to CSV File
    # DataF = DataFrame(time = time,
    #                   Vlvs  = Vlv,
    #                   lcs   = lc,
    #                   plvs  = plv,
    #                   parts = part  )
    # CSV.write("data/results_verification.csv", DataF)

    ## Plotting
    # PV-loop
    # display(plot(Vlv/ml, plv/mmHg, linewidth=3, label="Julia"))
    # #plot!(data_IGA[:Vlvs]/(ml^2), data_IGA[:plvs]/mmHg, linewidth=3, label="IGA")
    # #plot!(data_OF[:Vlvs]/ml, data_OF[:plvs]/mmHg, linewidth=3, label="Python")
    # xlabel!(L"$V^{\mathrm{lv}}$ [ml]")
    # ylabel!(L"$p^{\mathrm{lv}}$ [mmHg]")

    # Traces
    # p1 = plot(time/ms, [plv/mmHg, part/mmHg, Func.pven.(Vlv, part)/mmHg], linewidth=3, labels=[L"$p_{lv}$" L"$p_{art}$" L"$p_{ven}$"], ylabel=L"$p$  [mmHg]")
    # p2 = plot(time/ms, Vlv/ml, linewidth=3, legend=false, ylabel=L"$V_{lv}$ [ml]")
    # p3 = plot(time/ms, [Func.qper.(Vlv, part)/ml, Func.qart.(plv, part)/ml, Func.qven.(Vlv, plv, part)/ml], linewidth=3, labels=[L"q_{ven}" L"q_{art}" L"q_{ven}"], xlabel="Time [ms]", ylabel=L"$q_{flow}$ [ml/s]")
    # plot(p1, p2, p3, layout=(3,1))


    # plot(time/ms, Func.ls.(Vlv), linewidth=3, legend=false)
    # xlabel!(L"$t$ [ms]")
    # ylabel!(L"$l_{s}$ [μm]")
    
    # plot(time/ms, lc, linewidth=3, legend=false)
    # xlabel!(L"$t$ [ms]")
    # ylabel!(L"$l_{c}$ [μm]")
end




function main2()
    constants = Constants()
    (; ms, ml, kPa, mmHg) = constants

    # Initial values
    pven  = 1600   # Venous pressure at image config.
    Vlv0  = 44*ml  # Volume at 0 pressure
    Vlvi  = 120*ml # Volume at Image pressure 
    Vwall = 136*ml # Wall volume

    constants.Vlv0  = Vlv0
    constants.Vwall = Vwall

    # Starting point for GPA algorithm
    constants.tstart = 4*ms
    constants.trelax = 0*ms

    (; Vtotal, Vven0, Vart0, Cven, Cart) = constants
    pven = 1600
    part = ( Vtotal - Vlvi - Vven0 - Vart0 - Cven*pven ) / Cart

    initial   = Dict{String, Float64}("Vlv" => Vlvi, "lc" => 1.5, "plv" => pven, "part" => part)

    # Solve the time dependent problem
    (time, Vlv, lc, plv, part), Func = SolveOneFiber(constants, initial=initial, ncycles=3)


    ## Plotting
    # PV-loop
    display(plot(Vlv/ml, plv/mmHg, linewidth=3, label="Julia"))
    #plot!(data_IGA[:Vlvs]/(ml^2), data_IGA[:plvs]/mmHg, linewidth=3, label="IGA")
    xlabel!(L"$V^{\mathrm{lv}}$ [ml]")
    ylabel!(L"$p^{\mathrm{lv}}$ [mmHg]")

    # Traces
    # p1 = plot(time/ms, [plv/mmHg, part/mmHg, Func.pven.(Vlv, part)/mmHg], linewidth=3, labels=[L"$p_{lv}$" L"$p_{art}$" L"$p_{ven}$"], ylabel=L"$p$  [mmHg]")
    # p2 = plot(time/ms, Vlv/ml, linewidth=3, legend=false, ylabel=L"$V_{lv}$ [ml]")
    # p3 = plot(time/ms, [Func.qper.(Vlv, part)/ml, Func.qart.(plv, part)/ml, Func.qven.(Vlv, plv, part)/ml], linewidth=3, labels=[L"q_{ven}" L"q_{art}" L"q_{ven}"], xlabel="Time [ms]", ylabel=L"$q_{flow}$ [ml/s]")
    # plot(p1, p2, p3, layout=(3,1))


    # plot(time/ms, Func.ls.(Vlv), linewidth=3, legend=false)
    # xlabel!(L"$t$ [ms]")
    # ylabel!(L"$l_{s}$ [μm]")
    
    # plot(time/ms, lc, linewidth=3, legend=false)
    # xlabel!(L"$t$ [ms]")
    # ylabel!(L"$l_{c}$ [μm]")
end

main()

## Profiling commands
# using Profile, FlameGraphs
# @profile SolveOneFiber(constants, initial=initial, ncycles=6, return_eval=false, print_info=false)
# @profview SolveOneFiber(constants, initial=initial, ncycles=6, return_eval=false, print_info=false)