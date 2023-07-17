"""
    aDDM{T<:Real} <: AbstractaDDM
    
An object for the attentional diffusion model. 

# Parameters 

- `ν=[5.0,4.0]`: relative decision values (i.e., drift rates)
- `α=1.0`: evidence threshold 
- `z=0.0`: initial evidence 
- `θ=.3`: bias towards attended alternative (lower indicates more bias)
- `σ=.02`: standard deviation of noise in evidence accumulation
- `Δ=.0004`: constant of evidence accumulation speed (evidence per ms)
- `τ=0.0`: non-decision time

# Constructors

    aDDM(ν, α, z, θ, σ, Δ, τ)

    aDDM(;ν=[5.0,4.0], α=1.0, z=α*.5, θ=.3, σ=.02, Δ=.0004, τ=0.0)

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

 model = aDDM()
 
 tmat = Transition([.98 .015 .005;
                    .015 .98 .005;
                    .45 .45 .1])

 choices,rts = rand(model, 100, attend, tmat)
```
# References 

Krajbich, I., Armel, C., & Rangel, A. (2010). Visual fixations and the computation and comparison of 
value in simple choice. Nature neuroscience, 13(10), 1292-1298.
"""
struct aDDM{T<:Real} <: AbstractaDDM
    ν::Vector{T}
    α::T
    z::T
    θ::T
    σ::T
    Δ::T
    τ::T
end

function aDDM(ν, α, z, θ, σ, Δ, τ)
    _, α, z, θ, σ, Δ, τ = promote(ν[1], α, z, θ, σ, Δ, τ)
    ν = convert(Vector{typeof(z)}, ν)
    return aDDM(ν, α, z, θ, σ, Δ, τ)
end

function aDDM(;ν=[5.0,4.0], α=1.0, z=0.0, θ=0.3, σ=.02, Δ=.0004, τ=0.0)
    return aDDM(ν, α, z, θ, σ, Δ, τ)
end

get_pdf_type(d::aDDM) = Approximate


"""
    rand(rng::AbstractRNG, dist::AbstractaDDM, n_sim::Int, fixation, args...; rand_state! = _rand_state!, kwargs...)

Generate `n_sim` simulated trials from the attention diffusion model.

# Arguments

- `rng`: a random number generator
- `dist`: an attentional diffusion model object
- `n_sim::Int`: the number of simulated trials
- `fixation`: a function of the visual fixation process which returns 1 for alternative 
    and 2 for alternative 2
- `args...`: optional positional arguments for the `fixation` function

# Keywords
`rand_state! = _rand_state!`: initialize first state with equal probability 
- `kwargs...`: optional keyword arguments for the `fixation` function
"""
function rand(rng::AbstractRNG, dist::AbstractaDDM, n_sim::Int, fixation, args...; rand_state! = _rand_state!, kwargs...)
    choice = fill(0, n_sim)
    rt = fill(0.0, n_sim)
    for sim in 1:n_sim 
        rand_state!(rng, args...; kwargs...)
        choice[sim],rt[sim] = _rand(rng, dist, () -> fixation(args...; kwargs...))
    end
    return (;choice,rt)
end

_rand_state!(tmat) = _rand_state!(Random.default_rng(), tmat)

function _rand_state!(rng, tmat)
    tmat.state = rand(rng, 1:tmat.n)
    return nothing
 end
 
"""
    rand(rng::AbstractRNG, dist::AbstractaDDM, fixation, args...; kwargs...)

Generate a single simulated trial from the attention diffusion model.

# Arguments

- `rng`: a random number generator
- `dist`: an attentional diffusion model object
- `fixation`: a function of the visual fixation process which returns 1 for alternative 
    and 2 for alternative 2
- `args...`: optional positional arguments for the `fixation` function

# Keywords

- `kwargs...`: optional keyword arguments for the `fixation` function
"""
function rand(rng::AbstractRNG, dist::AbstractaDDM, fixation, args...; kwargs...)
    return _rand(rng, dist, () -> fixation(args...; kwargs...))
end

rand(dist::AbstractaDDM, fixation, args...; kwargs...) = rand(Random.default_rng(), dist, fixation, args...; kwargs...)

function _rand(rng::AbstractRNG, dist::AbstractaDDM, fixation)
    (;α,z,τ) = dist
    t = τ
    Δt = .001
    v = z
    while abs(v) < α
        t += Δt
        location = fixation()
        v += increment(rng, dist, location)
    end
    choice = (v < α) + 1
    return (;choice,rt=t)
end

increment(dist::AbstractaDDM, location) = increment(Random.default_rng(), dist, location)

"""
    increment(rng::AbstractRNG, dist::aDDM, location)

Returns the change evidence for a single iteration. 

# Arguments

- `rng`: a random number generator
- `dist::aDDM`: a model object for the attentional drift diffusion model
- `location`: an index for fixation location 
"""
function increment(rng::AbstractRNG, dist::aDDM, location)
    (;σ,ν,θ,Δ) = dist
    # option 1
    if location == 1
        return Δ * (ν[1] - θ * ν[2]) + noise(rng, σ)
    # option 2
    elseif location == 2
        return -Δ * (ν[2] - θ * ν[1]) + noise(rng, σ)
    else
        return noise(rng, σ)
    end
    return -100.0
end 

noise(rng, σ) = rand(rng, Normal(0, σ))

"""
    simulate(model::AbstractaDDM; fixation, m_args=(), m_kwargs=())

Returns a matrix containing evidence samples from a subtype of an attentional drift diffusion model decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::AbstractaDDM`: an drift diffusion  model object

# Keywords
- `attend`: a function of the visual fixation process which returns 1 for alternative 
    and 2 for alternative 2
- `args=()`: a set of optional positional arguments for the `attend` function 
- `kwargs=()`: a set of optional keyword arguments for the `attend` function 
`rand_state! = _rand_state!`: initialize first state with equal probability 
"""
function simulate(model::AbstractaDDM; attend, args=(), kwargs=(), rand_state! = _rand_state!)
    (;α,z) = model
    fixation = () -> attend(args...; kwargs...)
    rand_state!(args...)
    t = 0.0
    Δt = .001
    x = z
    evidence = [x]
    time_steps = [t]
    while abs(x) < α
        t += Δt
        location = fixation()
        x += increment(model, location)
        push!(evidence, x)
        push!(time_steps, t)
    end
    return time_steps,evidence
end