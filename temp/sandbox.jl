###################################################################################################
#                                        Load Packages
###################################################################################################
cd(@__DIR__)
using Pkg
using Revise
Pkg.activate("../")
using SequentialSamplingModels
using StatsBase
using SequentialSamplingModels: update
using SequentialSamplingModels: _rand_state!

model = maaDDM(ν₁₁ = 6.0, 
                ν₁₂ = 5.0, 
                ν₂₁ = 4.0, 
                ν₂₂ = 5.0, 
                α = 1.0, 
                z = 0.0, 
                θ = 0.3, 
                ϕ = 0.60, 
                ω = .50, 
                σ = .025, 
                Δ = .0004)

mutable struct Transition
    state::Int 
    n::Int
    mat::Array{Float64,2} 
 end

 function Transition(mat)
    n = size(mat,1)
    state = rand(1:n)
    return Transition(state, n, mat)
 end

 function attend(transition)
     (;mat,n,state) = transition
     w = mat[state,:]
     next_state = sample(1:n, Weights(w))
     transition.state = next_state
     return next_state
 end

 # transition within alternatives.
 # columns 
 # 1. alternative 1, attribute 1
 # 2. alternative 1, attribute 2
 # 3. alternative 2, attribute 1
 # 4. alternative 2, attribute 2
 tmat = Transition([.98 .02;
                    .02 .98])

rts = rand(model, 10_000, attend, tmat)

using Distributions 
using Plots 

sim_data = rand(Normal(0, 1), 10_000)

histogram(sim_data, norm=true, grid=false, leg=false)

x = range(-3, 3, length=100)
y = pdf.(Normal(0, 1), x)
plot!(x, y, color=:black, linewidth=2)

using KernelDensity
using Interpolations

sim_data = rand(MvNormal([0,0],[1 0;0 1]), 10_000)'
kd = kde(sim_data)
est_dist = InterpKDE(kd)

x = range(-3, 3, length=10)
v = mapreduce(y -> [y...], hcat, Base.product(x,x)) 
est_pdfs = [est_dist.itp(col...) for col in eachcol(v)]



using Revise, ProbabilityTheoryPlusNoiseModel, Random 
using InterferenceEffectUtilities
Random.seed!(3363)

ds = range(0, .5, length=50)
ns = fill(100, 3)

int_preds = map(d -> begin
                            dist = ObjectivePTN(; d)
                            preds = compute_preds(dist)
                            df = interference_effects(preds)
                            df.d .= d
                            return df
                     end, ds)

df = vcat(int_preds...)

pyplot()

@df df plot(:d, 
            :interference, 
            group = :condition,
            ylabel = "interference effect",
            xlabel = "d",
            grid = false, 
            xaxis = font(5),
            yaxis = font(5),
            legendfontsize = 4,
            linewidth = .75,
            ylims = (-.04,.04),
            size = (240,120),
            dpi = 300)

hline!([0.0], color = :black, linestyle = :dash, linewidth=.75, label=false)
