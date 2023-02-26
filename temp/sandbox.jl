###################################################################################################
#                                        Load Packages
###################################################################################################
cd(@__DIR__)
using Pkg
using Revise
Pkg.activate("../")
using SequentialSamplingModels
using StatsBase

model = maaDDM(ν₁₁ = 1.0, 
                ν₁₂ = 1.0, 
                ν₂₁ = 2.0, 
                ν₂₂ = 2.0, 
                α = 1.0, 
                z = 0.0, 
                θ = 0.0, 
                ϕ = 1.0, 
                ω = 1.0, 
                σ = .05, 
                Δ = .005)

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
 tmat = Transition([.98 .015 .0025 .0025;
                    .015 .98 .0025 .0025;
                    .0025 .0025 .98 .015;
                    .0025 .0025 .015 .98])

 rts = rand(model, 1000, attend, tmat)