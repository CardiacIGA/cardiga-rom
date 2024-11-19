using Plots
mmHg=133.322368

# Experimental results according to [10.1161/01.res.62.6.1210]
σi = 0
V0 = 37.2  + σi*5.6
Vm = 116.2 + σi*17.6
Vd = 13.1  + σi*2.0
Sp = 14.6  #+ σi*5.6
Sn = 5.1   #+ σi*5.6
Pp(V) = -Sp .* log( ( Vm - V  )/( Vm - V0 ) )*max(sign(V-V0), 0)
Pn(V) = -Sn .* log( ( V  - Vd )/( V0 - Vd ) )*min(sign(V-V0), 0)
P(V)  = Pp(V) + Pn(V)

V0    = V0  #60
Vwall = 180 #200
σf0   = 0.9*1e3
cf    = 12
σr0   = 0.2*1e3
cr    = 9

λf(V)  = ( ( V + 1/3*Vwall ) / ( V0 + 1/3*Vwall ) )^( 1 // 3 )
λr(V)  = λf(V)^(-2)
σmf(V) = σf0*( exp( cf*(λf(V)-1) ) - 1 )*( max(sign(λf(V) - 1), 0) )
σmr(V) = σr0*( exp( cr*(λr(V)-1) ) - 1 )*( max(sign(λr(V) - 1), 0) )
Pmodel(V) = 1/3*( σmf(V) - 2*σmr(V) )*log( 1 + (Vwall / V) )

Vs1 = range(Vd,110,100) # in [ml]
Vs2 = range(Vd,150,100)

plot(Vs1, P.(Vs1)*mmHg*1e-3)
plot!(Vs2, Pmodel.(Vs2)*1e-3)
ylims!(-3, 6)
xlims!(0, 150)