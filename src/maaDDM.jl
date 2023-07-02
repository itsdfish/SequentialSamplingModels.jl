"""
    maaDDM{T<:Real} <: AbstractaDDM

An object for the multi-attribute attentional drift diffusion model. 

# Constructors

    maaDDM(ν₁₁, ν₁₂, ν₂₁, ν₂₂, α, z, θ, ϕ, ω, σ, Δ)

    maaDDM(; ν₁₁ = 4.0, 
            ν₁₂ = 5.0, 
            ν₂₁ = 5.0, 
            ν₂₂ = 4.0, 
            α = 1.0, 
            z = 0.0, 
            θ = .3, 
            ϕ = .50, 
            ω = .70, 
            σ = .02, 
            Δ = .0004)
            
Constructor for multialternative attentional diffusion model object. 

In this version of the model, the non-attended attribute of the non-attended alternative is doubly discounted. For example,
the mean drift rate for the attribute 1 of alternative 1 is given by:

```julia
    Δ * (ω * (ν₁₁ - θ * ν₂₁) + (1 - ω) * ϕ * (ν₁₂ - θ * ν₂₂))
```

# Keywords 

- `ν₁₁=5.0`: relative decision value for alternative 1, attribute 1
- `ν₁₂=4.0`: relative decision value for alternative 1, attribute 2
- `ν₂₁=5.0`: relative decision value for alternative 2, attribute 1
- `ν₂₂=4.0`:  relative decision value for alternative 2, attribute 2
- `α=1.0`: evidence threshold 
- `z=0.0`: initial evidence 
- `θ=.3`: bias away from unattended alternative (lower indicates more bias)
- `ϕ=.50`: bias away from unattended attribute 
- `ω=.70`: attribute weight
- `σ=.02`: standard deviation of noise in evidence accumulation
- `Δ=.0004`: constant of evidence accumulation speed (evidence per ms)

# Example 

```julia 
using SequentialSamplingModels
using StatsBase

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

ν₁₁ = 4.0 
ν₁₂ = 5.0 
ν₂₁ = 5.0 
ν₂₂ = 4.0
α = 1.0 
z = 0.0
θ = .3
ϕ = .50
ω = .70
σ = .02
Δ = .0004

dist = maaDDM(; ν₁₁, ν₁₂, ν₂₁, ν₂₂, α, z, θ, ϕ, ω, σ, Δ)

tmat = Transition([.98 .015 .0025 .0025;
                .015 .98 .0025 .0025;
                .0025 .0025 .98 .015;
                .0025 .0025 .015 .98])

 choices,rts = rand(dist, 100, attend, tmat)
```

# References 

Yang, X., & Krajbich, I. (2023). A dynamic computational model of gaze and choice in multi-attribute decisions. 
Psychological Review, 130(1), 52.
"""
struct maaDDM{T<:Real} <: AbstractaDDM
    ν₁₁::T
    ν₁₂::T
    ν₂₁::T
    ν₂₂::T
    α::T
    z::T
    θ::T
    ϕ::T
    ω::T
    σ::T
    Δ::T
end

function maaDDM(ν₁₁, ν₁₂, ν₂₁, ν₂₂, α, z, θ, ϕ, ω, σ, Δ)
    return maaDDM(promote(ν₁₁, ν₁₂, ν₂₁, ν₂₂, α, z, θ, ϕ, ω, σ, Δ)...)
end

function maaDDM(; ν₁₁ = 4.0, 
                    ν₁₂ = 5.0, 
                    ν₂₁ = 5.0, 
                    ν₂₂ = 4.0, 
                    α = 1.0, 
                    z = 0.0, 
                    θ = .3, 
                    ϕ = .50, 
                    ω = .70, 
                    σ = .02, 
                    Δ = .0004)

    return maaDDM(ν₁₁, ν₁₂, ν₂₁, ν₂₂, α, z, θ, ϕ, ω, σ, Δ)
end

"""
    update(rng, dist::maaDDM, location)

Returns the change evidence for a single iteration. 

# Arguments

- `dist::maaDDM`: a model object for the multiattribute attentional drift diffusion model
- `location`: an index for fixation location 
"""
function update(rng, dist::maaDDM, location)
    (;ν₁₁,ν₁₂,ν₂₁,ν₂₂,θ,ϕ,ω,Δ,σ) = dist
    # option 1, attribute 1
    if location == 1
        return Δ * (ω * (ν₁₁ - θ * ν₂₁) + (1 - ω) * ϕ * (ν₁₂ - θ * ν₂₂)) + noise(rng, σ)
    # option 1, attribute 2
    elseif location == 2
        return Δ * (ϕ * ω * (ν₁₁ - θ * ν₂₁) + (1 - ω) * (ν₁₂ - θ * ν₂₂)) + noise(rng, σ)
    # option 2, attribute 1
    elseif location == 3
        return Δ * (ω * (θ * ν₁₁ - ν₂₁) + (1 - ω) * ϕ * (θ * ν₁₂ - ν₂₂)) + noise(rng, σ)
    # option 2, attribute 2
    else
        return Δ * (ϕ * ω * (θ * ν₁₁ - ν₂₁) + (1 - ω) * (θ * ν₁₂ - ν₂₂)) + noise(rng, σ)
    end
    return -100.0
end 