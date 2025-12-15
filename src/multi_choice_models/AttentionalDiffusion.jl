"""
    aDDM{T<:Real} <: AbstractaDDM
    
An object for the attentional diffusion model. 

# Parameters 

- `ν::Vector{T}`: relative decision values (i.e., drift rates). ν ∈ ℝⁿ.
- `σ::T`: standard deviation of noise in evidence accumulation. σ ∈ ℝ⁺.
- `Δ::T`: constant of evidence accumulation speed (evidence per ms). Δ ∈ ℝ⁺.
- `α::T`: evidence threshold. α ∈ ℝ⁺.  
- `z::T`: initial evidence. z ∈ [-α, α] 
- `θ::T`: bias towards attended alternative (lower indicates more bias) Δ ∈ ℝ⁺
- `τ::T`: non-decision time. τ ∈ [0, min_rt]

# Constructors

Two constructors are defined below. The first constructor uses positional arguments, and is therefore order dependent:


    aDDM(ν, σ, Δ, θ, α, z, τ)

The second constructor uses keywords with default values, and is not order dependent: 

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
 
 function fixate(transition)
     (;mat,n,state) = transition
     w = @view mat[state,:]
     next_state = sample(1:n, Weights(w))
     transition.state = next_state
     return next_state
 end

 model = aDDM()
 
 tmat = Transition([.98 .015 .005;
                    .015 .98 .005;
                    .45 .45 .1])

 choices,rts = rand(model, 100, tmat; fixate)
```
# References 

Krajbich, I., Armel, C., & Rangel, A. (2010). Visual fixations and the computation and comparison of 
value in simple choice. Nature neuroscience, 13(10), 1292-1298.
"""
struct aDDM{T <: Real} <: AbstractaDDM
    ν::Vector{T}
    σ::T
    Δ::T
    θ::T
    α::T
    z::T
    τ::T

    function aDDM(ν::Vector{T}, σ::T, Δ::T, θ::T, α::T, z::T, τ::T) where {T <: Real}
        @argcheck σ ≥ 0
        @argcheck Δ ≥ 0
        @argcheck θ ≥ 0
        @argcheck α ≥ 0
        @argcheck abs(z) ≤ α
        @argcheck τ ≥ 0
        return new{T}(ν, σ, Δ, θ, α, z, τ)
    end
end

function aDDM(ν, σ, Δ, θ, α, z, τ)
    _, σ, Δ, θ, α, z, τ = promote(ν[1], σ, Δ, θ, α, z, τ)
    ν = convert(Vector{typeof(z)}, ν)
    return aDDM(ν, σ, Δ, θ, α, z, τ)
end

function aDDM(; ν = [5.0, 4.0], α = 1.0, z = 0.0, θ = 0.3, σ = 0.02, Δ = 0.0004, τ = 0.0)
    return aDDM(ν, σ, Δ, θ, α, z, τ)
end

params(d::aDDM) = (d.ν, d.σ, d.Δ, d.θ, d.α, d.z, d.τ)

get_pdf_type(d::aDDM) = Approximate

"""
    rand(
        rng::AbstractRNG,
        dist::AbstractaDDM,
        n_sim::Int,
        fixate::Function, 
        args...;
        rand_state! = _rand_state!,
        Δt = .001,
        kwargs...
    )

Generate `n_sim` simulated trials from the attention diffusion model.

# Arguments

- `rng`: a random number generator
- `dist`: an attentional diffusion model object
- `n_sim::Int`: the number of simulated trials
- `fixate`: a function of the visual fixation process which returns 1 for alternative 
    and 2 for alternative 2
- `args...`: optional positional arguments for the `fixate` function

# Keywords
`rand_state! = _rand_state!`: initialize first state with equal probability 
- `kwargs...`: optional keyword arguments for the `fixate` function
- ` Δt = .001`: time step
"""
function rand(
    rng::AbstractRNG,
    dist::AbstractaDDM,
    n_sim::Int,
    args...;
    fixate,
    rand_state! = _rand_state!,
    Δt = 0.001,
    kwargs...
)
    choice = fill(0, n_sim)
    rt = fill(0.0, n_sim)
    for sim = 1:n_sim
        rand_state!(rng, args...; kwargs...)
        choice[sim], rt[sim] =
            rand(rng, dist; fixate = () -> fixate(args...; kwargs...), Δt)
    end
    return (; choice, rt)
end

_rand_state!(tmat) = _rand_state!(Random.default_rng(), tmat)

function _rand_state!(rng, tmat)
    tmat.state = rand(rng, 1:(tmat.n))
    return nothing
end

"""
    rand(
        rng::AbstractRNG, 
        dist::AbstractaDDM, 
        fixate::Function, args...; 
        rand_state! = _rand_state!, 
        Δt = .001,
        kwargs...
    )

Generate a single simulated trial from the attentional diffusion model.

# Arguments

- `rng`: a random number generator
- `dist`: an attentional diffusion model object
- `fixate`: a function of the visual fixation process which returns 1 for alternative 
    and 2 for alternative 2
- `args...`: optional positional arguments for the `fixate` function

# Keywords

- `kwargs...`: optional keyword arguments for the `fixate` function
- `Δt = .001`: time step
"""
function rand(
    rng::AbstractRNG,
    dist::AbstractaDDM,
    args...;
    fixate,
    rand_state! = _rand_state!,
    Δt = 0.001,
    kwargs...
)
    rand_state!(rng, args...; kwargs...)
    return rand(rng, dist; fixate = () -> fixate(args...; kwargs...), Δt)
end

function rand(dist::AbstractaDDM, args...; fixate, Δt = 0.001, kwargs...)
    return rand(Random.default_rng(), dist::AbstractaDDM, args...; fixate, Δt, kwargs...)
end

function rand(dist::AbstractaDDM, n_sim::Int, args...; fixate, Δt = 0.001, kwargs...)
    return rand(Random.default_rng(), dist, n_sim, args...; fixate, Δt, kwargs...)
end

function rand(rng::AbstractRNG, dist::AbstractaDDM; fixate, Δt = 0.001)
    (; α, z, τ) = dist
    t = τ
    v = z
    while abs(v) < α
        t += Δt
        location = fixate()
        v += increment!(rng, dist, location)
    end
    choice = (v < α) + 1
    return (; choice, rt = t)
end

"""
    cdf(
        rng::AbstractRNG, 
        d::AbstractaDDM, 
        choice::Int, 
        fixate::Function, 
        ub, 
        args...; 
        n_sim=10_000, 
        kwargs...
    )

Computes the approximate cumulative probability density of `AbstractaDDM` using Monte Carlo simulation.

# Arguments

- `dist`: an attentional diffusion model object
- `choice`: the choice on which the cumulative density is computed
- `fixate`: a function of the visual fixation process which returns 1 for alternative 
    and 2 for alternative 2
- `ub::Int`: the upper bound of the integral
- `args...`: optional positional arguments for the `fixate` function

# Keywords

- `n_sim::Int`=10_000: the number of simulated trials
`rand_state! = _rand_state!`: initialize first state with equal probability 
- `kwargs...`: optional keyword arguments for the `fixate` function
"""
function cdf(
    rng::AbstractRNG,
    d::AbstractaDDM,
    choice::Int,
    ub,
    args...;
    fixate,
    n_sim = 10_000,
    kwargs...
)
    c, rt = rand(rng, d, n_sim, args...; fixate, kwargs...)
    return mean(c .== choice .&& rt .≤ ub)
end

function cdf(d::AbstractaDDM, choice::Int, ub::Real, args...; fixate, kwargs...)
    return cdf(Random.default_rng(), d, choice, ub, args...; fixate, kwargs...)
end

function survivor(
    rng::AbstractRNG,
    d::AbstractaDDM,
    choice::Int,
    ub,
    args...;
    fixate::Function,
    n_sim = 10_000,
    kwargs...
)
    return 1 - cdf(rng, d, choice, ub, args...; fixate, kwargs...)
end

function survivor(d::AbstractaDDM, choice::Int, ub::Real, args...; fixate, kwargs...)
    return survivor(Random.default_rng(), d, choice, fixate, ub, args...; fixate, kwargs...)
end

increment!(dist::AbstractaDDM, location) = increment!(Random.default_rng(), dist, location)

"""
    increment!(rng::AbstractRNG, dist::aDDM, location)

Returns the change evidence for a single iteration. 

# Arguments

- `rng`: a random number generator
- `dist::aDDM`: a model object for the attentional drift diffusion model
- `location`: an index for fixation location 
"""
function increment!(rng::AbstractRNG, dist::aDDM, location)
    (; σ, ν, θ, Δ) = dist
    # option 1
    if location == 1
        return Δ * (ν[1] - θ * ν[2]) + noise(rng, σ)
        # option 2
    elseif location == 2
        return -Δ * (ν[2] - θ * ν[1]) + noise(rng, σ)
    end
    return noise(rng, σ)
end

noise(rng, σ) = rand(rng, Normal(0, σ))

"""
    simulate(
        rng::AbstractRNG, 
        model::AbstractaDDM; 
        fixate, 
        args=(), 
        kwargs=(), 
        Δt = .001,
        rand_state! = _rand_state!
    )

Returns a matrix containing evidence samples from a subtype of an attentional drift diffusion model decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `rng::AbstractRNG`: random number generator 
- `model::AbstractaDDM`: an drift diffusion  model object

# Keywords

- `fixate`: a function of the visual fixation process which returns 1 for alternative 
    and 2 for alternative 2
- `args=()`: a set of optional positional arguments for the `attend` function 
- `kwargs=()`: a set of optional keyword arguments for the `attend` function 
- `Δt = .001`: time step 
`rand_state! = _rand_state!`: initialize first state with equal probability 
"""
function simulate(
    rng::AbstractRNG,
    model::AbstractaDDM,
    args...;
    fixate,
    Δt = 0.001,
    rand_state! = _rand_state!,
    kwargs...
)
    (; α, z) = model
    _fixate = () -> fixate(args...; kwargs...)
    rand_state!(args...)
    t = 0.0
    x = z
    evidence = [x]
    time_steps = [t]
    while abs(x) < α
        t += Δt
        location = _fixate()
        x += increment!(model, location)
        push!(evidence, x)
        push!(time_steps, t)
    end
    return time_steps, evidence
end
