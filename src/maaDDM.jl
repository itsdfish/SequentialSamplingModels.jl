"""
    maaDDM{T<:Real} <: AbstractaDDM

An object for the multi-attribute attentional drift diffusion model. 

# Constructors

    maaDDM(ν, α, z, θ, ϕ, ω, σ, Δ, τ)

    maaDDM(; 
        ν = [4.0 5.0; 5.0 4.0],
        α = 1.0, 
        z = 0.0, 
        θ = .3, 
        ϕ = .50, 
        ω = .70, 
        σ = .02, 
        Δ = .0004,
        τ = 0.0)
            
Constructor for multialternative attentional diffusion model object. 

In this version of the model, the non-attended attribute of the non-attended alternative is doubly discounted. For example,
the mean drift rate for the attribute 1 of alternative 1 is given by:

```julia
    Δ * (ω * (ν[1,1] - θ * ν[2,1]) + (1 - ω) * ϕ * (ν[1,2] - θ * ν[2,2]))
```

# Keywords 
- `ν`: drift rates where rows are alternatives and columns are attributes
- `α`: evidence threshold 
- `z`: initial evidence 
- `θ`: bias away from unattended alternative (lower indicates more bias)
- `ϕ`: bias away from unattended attribute 
- `ω`: attribute weight
- `σ`: standard deviation of noise in evidence accumulation
- `Δ`: constant of evidence accumulation speed (evidence per ms)
- `τ`: non-decision time

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

ν = [4.0 5.0; 5.0 4.0]
α = 1.0 
z = 0.0
θ = .3
ϕ = .50
ω = .70
σ = .02
Δ = .0004
τ = 0.0

dist = maaDDM(; ν, α, z, θ, ϕ, ω, σ, Δ, τ)

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
    ν::Array{T,2}
    α::T
    z::T
    θ::T
    ϕ::T
    ω::T
    σ::T
    Δ::T
    τ::T
end

function maaDDM(ν, α, z, θ, ϕ, ω, σ, Δ, τ)
    _, α, z, θ, ϕ, ω, σ, Δ, τ = promote(ν[1], α, z, θ, ϕ, ω, σ, Δ, τ)
    ν = convert(Array{typeof(z),2}, ν)
    return maaDDM(ν, α, z, θ, ϕ, ω, σ, Δ, τ)
end

function maaDDM(; 
    ν = [4.0 5.0; 5.0 4.0],
    α = 1.0, 
    z = 0.0, 
    θ = .3, 
    ϕ = .50, 
    ω = .70, 
    σ = .02, 
    Δ = .0004,
    τ = 0.0)

    return maaDDM(ν, α, z, θ, ϕ, ω, σ, Δ, τ)
end

get_pdf_type(d::maaDDM) = Approximate

n_options(d::maaDDM) = size(d.ν, 1)

"""
    increment(rng, dist::maaDDM, location)

Returns the change evidence for a single iteration. 

# Arguments

- `dist::maaDDM`: a model object for the multiattribute attentional drift diffusion model
- `location`: an index for fixation location 
"""
function increment(rng, dist::maaDDM, location)
    (;ν,θ,ϕ,ω,Δ,σ) = dist
    # option 1, attribute 1
    if location == 1
        return Δ * (ω * (ν[1,1] - θ * ν[2,1]) + (1 - ω) * ϕ * (ν[1,2] - θ * ν[2,2])) + noise(rng, σ)
    # option 1, attribute 2
    elseif location == 2
        return Δ * (ϕ * ω * (ν[1,1] - θ * ν[2,1]) + (1 - ω) * (ν[1,2] - θ * ν[2,2])) + noise(rng, σ)
    # option 2, attribute 1
    elseif location == 3
        return Δ * (ω * (θ * ν[1,1] - ν[2,1]) + (1 - ω) * ϕ * (θ * ν[1,2] - ν[2,2])) + noise(rng, σ)
    # option 2, attribute 2
    else
        return Δ * (ϕ * ω * (θ * ν[1,1] - ν[2,1]) + (1 - ω) * (θ * ν[1,2] - ν[2,2])) + noise(rng, σ)
    end
    return -100.0
end 