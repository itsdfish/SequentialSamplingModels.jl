"""
    stDDM{T<:Real} <: AbstractstDDM

An object for the starting-time diffusion decision model.

# Parameters 

- `ν::T`:  vector of drift rate weights for attribute one and two. ν ∈ ℝ.
- `σ::T`:  diffusion noise. σ ∈ ℝ⁺.
- `η::T`:  vector of variability in drift rate for attribute one and two. η ∈ ℝ⁺.
- `s::T`:  initial latency bias. s ∈ ℝ⁺. (-s for attribute 1, s for attribute 2)
- `ρ::T`:  correlation between drift rate for attributes. ρ ∈ [-1,1].
- `α::T`:  evidence threshold. α ∈ ℝ⁺.
- `z::T`:  initial evidence. z ∈ [0,α].  
- `τ::T`:  non-decision time. τ ∈ [0, min_rt].

# Constructors 

    stDDM(ν, σ, η, s, ρ, α, z, τ)

    stDDM(;
        ν = [0.5, 0.6],
        σ = 1,
        s = 0.50,
        ρ = 0.0,
        η = fill(1.0, length(ν)),
        α = 1.0,
        z = 0.50,
        τ = 0.300
    )
        
# Example 

```julia 
using SequentialSamplingModels

ν = [0.5, 0.6]
σ = 1
η = [1.0, 1.0]
ρ = 0.00
s = 0.50
α = 1.0
z = 0.50
τ = 0.300

# Create stDDM model instance
dist = stDDM(; ν, σ, s, z, η, ρ, α, τ)

choices,rts = rand(dist, 500)
```

# References

Amasino, D.R., Sullivan, N.J., Kranton, R.E. et al. Amount and time exert independent influences on intertemporal choice. Nat Hum Behav 3, 383–392 (2019). https://doi.org/10.1038/s41562-019-0537-2

Barakchian, Z., Beharelle, A.R. & Hare, T.A. Healthy decisions in the cued-attribute food choice paradigm have high test-retest reliability. Sci Rep, (2021). https://doi.org/10.1038/s41598-021-91933-6

Chen, HY., Lombardi, G., Li, SC. et al. Older adults process the probability of winning sooner but weigh it less during lottery decisions. Sci Rep, (2022). https://doi.org/10.1038/s41598-022-15432-y

Lombardi, G., & Hare, T. Piecewise constant averaging methods allow for fast and accurate hierarchical Bayesian estimation of drift diffusion models with time-varying evidence accumulation rates. PsyArXiv, (2021). https://doi.org/10.31234/osf.io/5azyx

Sullivan, N.J., Huettel, S.A. Healthful choices depend on the latency and rate of information accumulation. Nat Hum Behav 5, 1698–1706 (2021). https://doi.org/10.1038/s41562-021-01154-0
"""

mutable struct stDDM{T <: Real} <: AbstractstDDM
    ν::Vector{T}
    σ::T
    η::Vector{T}
    s::T
    ρ::T
    α::T
    z::T
    τ::T
    function stDDM(
        ν::Vector{T},
        σ::T,
        η::Vector{T},
        s::T,
        ρ::T,
        α::T,
        z::T,
        τ::T
    ) where {T <: Real}
        @argcheck length(ν) == length(η)
        @argcheck σ ≥ 0
        @argcheck all(η .≥ 0)
        @argcheck s ≥ 0
        @argcheck (ρ ≥ -1) && (ρ ≤ 1)
        @argcheck α ≥ 0
        @argcheck (z ≥ 0) && (z ≤ α)
        @argcheck τ ≥ 0
        return new{T}(ν, σ, η, s, ρ, α, z, τ)
    end
end

function stDDM(ν, σ, η, s, ρ, α, z, τ::T) where {T}
    _, σ, _, s, ρ, α, z, τ = promote(ν[1], σ, η[1], s, ρ, α, z, τ)
    ν = convert(Vector{T}, ν)
    η = convert(Vector{T}, η)
    return stDDM(ν, σ, η, s, ρ, α, z, τ)
end

function stDDM(;
    ν = [0.5, 0.6],
    σ = 1,
    s = 0.50,
    ρ = 0.0,
    η = fill(1.0, length(ν)),
    α = 1.0,
    z = 0.50,
    τ = 0.300
)
    return stDDM(ν, σ, η, s, ρ, α, z, τ)
end

params(d::AbstractstDDM) = (d.ν, d.σ, d.η, d.s, d.ρ, d.α, d.z, d.τ)

get_pdf_type(::AbstractstDDM) = Approximate

"""
    rand(dist::AbstractstDDM)

Generate a random choice-rt pair for starting-time diffusion decision model.

# Arguments
- `rng`: a random number generator
- `dist`: model object for the starting-time diffusion decision model. 
- `Δt`: time-step for simulation
"""
function rand(rng::AbstractRNG, dist::AbstractstDDM; kwargs...)
    return simulate_trial(rng, dist; kwargs...)
end

"""
    simulate_trial(rng::AbstractRNG, dist::AbstractstDDM;  Δt, max_steps)

Generate a single simulated trial from the starting-time diffusion decision model.

# Arguments

- `rng`: a random number generator
- `model::AbstractstDDM`: a starting-time diffusion decision model object
- `Δt`: time-step for simulation
- `max_steps`: total/max time for simulation
"""
function simulate_trial(rng::AbstractRNG, d::AbstractstDDM; Δt = 0.001, max_steps = 6)
    (; ν, σ, s, z, η, ρ, α, τ) = d

    lt = Int(max_steps / Δt)
    start_step = abs(Int(s / Δt))

    t = τ
    choice = 0  # Initialize choice with a default value

    X = z * α
    deciding = true
    cont = 1
    Ρ = [1.0 ρ; ρ 1.0]
    Σ = cor2cov(Ρ, η)
    ν₁, ν₂ = rand(MvNormal(ν, Σ))

    while deciding && cont <= lt
        δ1 = cont ≤ start_step && s > 0 ? 0.0 : 1.0
        δ2 = cont ≤ start_step && s < 0 ? 0.0 : 1.0

        noise = rand(rng, Normal(0, σ)) * √(Δt)

        X += (ν₁ * δ1 + ν₂ * δ2) * Δt + noise
        if X > α
            choice = 1
            deciding = false
        elseif X < 0
            choice = 2
            deciding = false
        end
        t += Δt

        cont += 1
    end

    return (; choice, rt = t)
end

"""
    simulate(rng::AbstractRNG, model::AbstractstDDM; Δt)

Returns a matrix containing evidence samples of the stDDM decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `rng`: a random number generator
- `model::AbstractstDDM`: a starting-time diffusion decision model diffusion model object
- `Δt`: time-step for simulation
"""
function simulate(rng::AbstractRNG, model::AbstractstDDM; Δt = 0.001)
    (; ν, σ, s, z, η, ρ, α, τ) = model

    x = α * z
    t = 0.0
    evidence = [x]
    time_steps = [t]
    cont = 1
    start_step = abs(Int(s / Δt))

    Ρ = [1.0 ρ; ρ 1.0]
    Σ = cor2cov(Ρ, η)
    ν₁, ν₂ = rand(MvNormal(ν, Σ))
    while (x < α) && (x > 0)
        t += Δt

        δ1 = cont ≤ start_step && s > 0 ? 0.0 : 1.0
        δ2 = cont ≤ start_step && s < 0 ? 0.0 : 1.0

        noise = rand(rng, Normal(0, σ)) * √(Δt)

        x += (ν₁ * δ1 + ν₂ * δ2) * Δt + noise

        push!(evidence, x)
        push!(time_steps, t)
        cont += 1
    end

    return time_steps, evidence
end
