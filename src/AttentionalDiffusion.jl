"""
    aDDM{T<:Real} <: AbstractaDDM
    
An object for the attentional diffusion model. 

# Parameters 

- `ν1=5.0`: relative decision value for alternative 1
- `ν2=4.0`: relative decision value for alternative 2
- `α=1.0`: evidence threshold 
- `z=0.0`: initial evidence 
- `θ=.3`: bias towards attended alternative (lower indicates more bias)
- `σ=.02`: standard deviation of noise in evidence accumulation
- `Δ=.0004`: constant of evidence accumulation speed (evidence per ms)

# Constructors

    aDDM(ν1, ν2, α, z, θ, σ, Δ)

    aDDM(;ν1=5.0, ν2=4.0, α=1.0, z=α*.5, θ=.3, σ=.02, Δ=.0004)

# References 

Krajbich, I., Armel, C., & Rangel, A. (2010). Visual fixations and the computation and comparison of 
value in simple choice. Nature neuroscience, 13(10), 1292-1298.
"""
struct aDDM{T<:Real} <: AbstractaDDM
    ν1::T
    ν2::T
    α::T
    z::T
    θ::T
    σ::T
    Δ::T
end

function aDDM(ν1, ν2, α, z, θ, σ, Δ)
    return aDDM(promote(ν1, ν2, α, z, θ, σ, Δ)...)
end

function aDDM(;ν1=5.0, ν2=4.0, α=1.0, z=0.0, θ=.3, σ=.02, Δ=.0004)
    return aDDM(ν1, ν2, α, z, θ, σ, Δ)
end

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
    (;α,z) = dist
    t = 0.0
    Δt = .001
    v = z
    while abs(v) < α
        t += Δt
        location = fixation()
        v += update(rng, dist, location)
    end
    choice = (v < α) + 1
    return (;choice,rt=t)
end

"""
    update(rng::AbstractRNG, dist::aDDM, location)

Returns the change evidence for a single iteration. 

# Arguments

- `rng`: a random number generator
- `dist::aDDM`: a model object for the attentional drift diffusion model
- `location`: an index for fixation location 
"""
function update(rng::AbstractRNG, dist::aDDM, location)
    (;σ,ν1,ν2,θ,Δ) = dist
    # option 1
    if location == 1
        return Δ * (ν1 - θ * ν2) + noise(rng, σ)
    # option 2
    elseif location == 2
        return -Δ * (ν2 - θ * ν1) + noise(rng, σ)
    else
        return noise(rng, σ)
    end
    return -100.0
end 

update(dist::AbstractaDDM, location) = update(Random.default_rng(), dist, location)

noise(rng, σ) = rand(rng, Normal(0, σ))