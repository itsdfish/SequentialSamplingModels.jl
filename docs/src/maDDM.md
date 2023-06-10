 ### Multiattribute Attentional Drift Diffusion

 ```julia 
using StatsBase

model = maaDDM()

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

 choice,rts = rand(model, 1000, attend, tmat)
 ```