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



function full(pA, pAgBC, pAgnBC, pBC, d)
    v1 = 1 - 2 * d 
    t1 = v1 * pAgBC + d 
    t2 = (1 - d) * pBC
    t3 = v1 * pAgnBC + d
    t4 = d * (1 - pBC)
    t5 = v1 * pBC + d
    return (t1 * t2 + t3 * t4) / t5
end



function compute_a_given_bc(pA, pBC, pABC, d)
    v1 = 1 - 2 * d 
    t1 = v1 * (pABC - d * (pA - 2 * pABC))
    t2 = d * (v1 * pBC + d)
    t3 = v1 * pBC + d
    return (t1 + t2) / t3
end

# function compute_a_given_bc1(pA, pBC, pABC, d)
#     v1 = 1 - 2 * d 
#     t1 = v1 * (pABC - d * (pA - 2 * pABC + pBC)) + d^2
#     t2 = v1 * pBC + d
#     return t1 / t2
# end

d = .1
dist = ObjectivePTN(; d)
pA = sum(dist.probs[1,:,:])
pB = sum(dist.probs[:,1,:])
pBC = sum(dist.probs[:,1,1])
pABC = dist.probs[1,1,1]
pAgBC = dist.probs[1,1,1] / sum(dist.probs[:,1,1])
pAgnBC = (sum(dist.probs[1,:,:]) - dist.probs[1,1,1])/ (1 - sum(dist.probs[:,1,1]))
compute_a_given_bc(pA, pBC, pABC, d)
full(pA, pAgBC, pAgnBC, pBC, d)
#conditional_prob(p_A, p_B, p_AB, d)
# probability attack given b-type and bad 
ProbabilityTheoryPlusNoiseModel.compute_a_given_b(pA, pBC, pABC, d)


((1 - 2 * d) * pAgBC + d) * (1 - d) * pBC + ((1 - 2 * d) * pAgnBC + d) * d * (1 - pBC)

# (1 - d) * pBC * (1 - 2 * d) * pAgBC + d * (1 - d) * pBC  + d * (1 - pBC) * (1 - 2 * d) * pAgnBC + d * d * (1 - pBC)


(1 - d) * (1 - 2 * d) * pABC + d * (1 - d) * pBC + d * (1 - 2 * d) * (pA - pABC) + d^2 * (1 - pBC)

# group

(1 - 2 * d) * ((1 - d) * pABC + d * (pA - pABC)) +  d *((1 - d) * pBC + d * (1 - pBC))

# distribute 
(1 - 2 * d) * (pABC - d * pABC + d * pA - d * pABC) + d * (pBC - d * pBC + d - d * pBC)

(1 - 2 * d) * (pABC + d * (pA - 2 * pABC)) + d *((1 - 2 * d) * pBC + d)

(1 - 2 * d) * ((pABC + d * (pA - 2 * pABC)) + d * pBC)  + d^2