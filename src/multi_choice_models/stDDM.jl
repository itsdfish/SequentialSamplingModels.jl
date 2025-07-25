"""
    stDDM{T<:Real} <: AbstractstDDM

An object for the starting-time diffusion decision model.

# Parameters 

- `ν::T`:  vector of drift rate weights for attribute one and two. ν ∈ ℝ.
- `σ::T`:  diffusion noise. σ ∈ ℝ⁺.
- `s::T`:  initial latency bias (positive for attribute two, negative for attribute one)
- `z::T`:  initial evidence. z ∈ [0,α].  
- `η::T`:  vector of variability in drift rate for attribute one and two. η ∈ ℝ⁺.
- `ρ::T`:  correlation between drift rate for attributes. ρ ∈ [-1,1].
- `α::T`:  evidence threshold. α ∈ ℝ⁺.
- `τ::T`:  non-decision time. τ ∈ [0, min_rt].

# Constructors 

    stDDM(ν, σ, s, z, η, ρ, α, τ)

    stDDM(;ν = [0.5,0.6],σ = 1,s = 0.50, z = 0.50, η = [1.0,1.0], ρ = 0.00, α = 1.0, τ = .300)
        
# Example 

```julia 
using SequentialSamplingModels

ν = [0.5, 0.6]
σ = 1
s = 0.50
z = 0.50
η = [1.0, 1.0]
ρ = 0.00
α = 1.0
τ = 0.300

# Create stDDM model instance
dist = stDDM(;ν, σ, s, z, η, ρ, α, τ)

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
    s::T
    z::T
    η::Vector{T}
    ρ::T
    α::T
    τ::T
end

function stDDM(ν, σ, s, z, η, ρ, α, τ::T) where {T}
    _, σ, s, z, _, ρ, α, τ = promote(ν[1], σ, s, z, η[1], ρ, α, τ)
    ν = convert(Vector{T}, ν)
    η = convert(Vector{T}, η)
    return stDDM(ν, σ, s, z, η, ρ, α, τ)
end

function stDDM(;
    ν = [0.5, 0.6],
    σ = 1,
    s = 0.50,
    z = 0.50,
    η = fill(1.0, length(ν)),
    ρ = 0.0,
    α = 1.0,
    τ = 0.300
)
    return stDDM(ν, σ, s, z, η, ρ, α, τ)
end

function params(d::AbstractstDDM)
    (d.ν, d.σ, d.s, d.z, d.η, d.ρ, d.α, d.τ)
end

get_pdf_type(d::AbstractstDDM) = Approximate

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
