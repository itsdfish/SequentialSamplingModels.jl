"""
    LBA{T <: Real, T1 <: Union{<: T, Vector{<: T}}} <: AbstractLBA

A model object for the linear ballistic accumulator.

# Parameters

- `ν::Vector{T}`: a vector of drift rates. α ∈ ℝ⁺.
- `σ::T1`: a scalar or vector of drift rate standard deviation. σ ∈ ℝ⁺.
- `A::T`: max start point. A ∈ ℝ⁺.
- `k::T`: A + k = b, where b is the decision threshold. k ∈ ℝ⁺.
- `τ::T`: an encoding-response offset. τ ∈ [0, min_rt].

# Constructors 

Two constructors are defined below. The first constructor uses positional arguments, and is therefore order dependent:

    LBA(ν, σ, A, k, τ)

The second constructor uses keywords with default values, and is not order dependent: 

    LBA(;τ = .3, A = .8, k = .5, ν = [2.0,1.75], σ = 1)

# Example 

```julia
using SequentialSamplingModels
dist = LBA(ν=[3.0,2.0], A = .8, k = .2, τ = .3) 
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
```

# References

Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice response time: Linear ballistic accumulation. Cognitive psychology, 57(3), 153-178.
"""
mutable struct LBA{T <: Real, T1 <: Union{<:T, Vector{<:T}}} <: AbstractLBA{T, T1}
    ν::Vector{T}
    σ::T1
    A::T
    k::T
    τ::T
end

function LBA(ν, σ, A, k, τ::T) where {T}
    _, _, A, k, τ = promote(ν[1], σ[1], A, k, τ)
    ν = convert(Vector{T}, ν)
    σ = isa(σ, Vector) ? convert(Vector{T}, σ) : convert(T, σ)
    return LBA(ν, σ, A, k, τ)
end

LBA(; τ = 0.3, A = 0.8, k = 0.5, ν = [2.0, 1.75], σ = 1) =
    LBA(ν, σ, A, k, τ)

function params(d::LBA)
    return (d.ν, d.σ, d.A, d.k, d.τ)
end

function select_winner(dt)
    if any(x -> x > 0, dt)
        mi, mv = 0, Inf
        for (i, t) in enumerate(dt)
            if (t > 0) && (t < mv)
                mi = i
                mv = t
            end
        end
    else
        return 1, -1.0
    end
    return mi, mv
end

sample_drift_rates(ν, σ) = sample_drift_rates(Random.default_rng(), ν, σ)

function sample_drift_rates(rng::AbstractRNG, ν, σ)
    negative = true
    v = similar(ν)
    n_options = length(ν)
    while negative
        v = @. rand(rng, Normal(ν, σ))
        negative = any(x -> x > 0, v) ? false : true
    end
    return v
end

function rand(rng::AbstractRNG, d::AbstractLBA)
    (; τ, A, k, ν, σ) = d
    b = A + k
    N = length(ν)
    v = sample_drift_rates(rng, ν, σ)
    a = rand(rng, Uniform(0, A), N)
    dt = @. (b - a) / v
    choice, mn = select_winner(dt)
    rt = τ .+ mn
    return (; choice, rt)
end

function logpdf(d::AbstractLBA{T, T1}, c, rt) where {T, T1 <: Vector{<:Real}}
    (; τ, A, k, ν, σ) = d
    b = A + k
    LL = 0.0
    rt < τ ? (return -Inf) : nothing
    for i ∈ 1:length(ν)
        if c == i
            LL += log_dens(d, ν[i], σ[i], rt)
        else
            LL += log(max(0.0, 1 - cummulative(d, ν[i], σ[i], rt)))
        end
    end
    pneg = pnegative(d)
    LL = LL - log(1 - pneg)
    return max(LL, -1000.0)
end

function logpdf(d::AbstractLBA{T, T1}, c, rt) where {T, T1 <: Real}
    (; τ, A, k, ν, σ) = d
    b = A + k
    LL = 0.0
    rt < τ ? (return -Inf) : nothing
    for i ∈ 1:length(ν)
        if c == i
            LL += log_dens(d, ν[i], σ, rt)
        else
            LL += log(max(0.0, 1 - cummulative(d, ν[i], σ, rt)))
        end
    end
    pneg = pnegative(d)
    LL = LL - log(1 - pneg)
    return max(LL, -1000.0)
end

function pdf(d::AbstractLBA{T, T1}, c, rt) where {T, T1 <: Vector{<:Real}}
    (; τ, A, k, ν, σ) = d
    b = A + k
    den = 1.0
    rt < τ ? (return 1e-10) : nothing
    for i ∈ 1:length(ν)
        if c == i
            den *= dens(d, ν[i], σ[i], rt)
        else
            den *= (1 - cummulative(d, ν[i], σ[i], rt))
        end
    end
    pneg = pnegative(d)
    den = den / (1 - pneg)
    den = max(den, 1e-10)
    isnan(den) ? (return 0.0) : (return den)
end

function pdf(d::AbstractLBA{T, T1}, c, rt) where {T, T1 <: Real}
    (; τ, A, k, ν, σ) = d
    b = A + k
    den = 1.0
    rt < τ ? (return 1e-10) : nothing
    for i ∈ 1:length(ν)
        if c == i
            den *= dens(d, ν[i], σ, rt)
        else
            den *= (1 - cummulative(d, ν[i], σ, rt))
        end
    end
    pneg = pnegative(d)
    den = den / (1 - pneg)
    den = max(den, 1e-10)
    isnan(den) ? (return 0.0) : (return den)
end

function dens(d::AbstractLBA, v, σ, rt)
    (; τ, A, k) = d
    dt = rt - τ
    b = A + k
    n1 = (b - A - dt * v) / (dt * σ)
    n2 = (b - dt * v) / (dt * σ)
    dens = (1 / A) * (-v * Φ(n1) + σ * ϕ(n1) + v * Φ(n2) - σ * ϕ(n2))
    return max(dens, 0.0)
end

function log_dens(d::AbstractLBA, v, σ, rt)
    (; τ, A, k) = d
    dt = rt - τ
    b = A + k
    n1 = (b - A - dt * v) / (dt * σ)
    n2 = (b - dt * v) / (dt * σ)
    dens = -log(A) + log(max(0.0, -v * Φ(n1) + σ * ϕ(n1) + v * Φ(n2) - σ * ϕ(n2)))
    return dens
end

function cummulative(d::AbstractLBA, v, σ, rt)
    (; τ, A, k) = d
    dt = rt - τ
    b = A + k
    n1 = (b - A - dt * v) / (dt * σ)
    n2 = (b - dt * v) / (dt * σ)
    cm =
        1 + ((b - A - dt * v) / A) * Φ(n1) - ((b - dt * v) / A) * Φ(n2) +
        ((dt * σ) / A) * ϕ(n1) - ((dt * σ) / A) * ϕ(n2)
    return cm
end

function pnegative(d::AbstractLBA{T, T1}) where {T, T1 <: Vector{<:Real}}
    (; ν, σ) = d
    p = 1.0
    for i ∈ 1:length(ν)
        p *= Φ(-ν[i] / σ[i])
    end
    return p
end

function pnegative(d::AbstractLBA{T, T1}) where {T, T1 <: Real}
    (; ν, σ) = d
    p = 1.0
    for i ∈ 1:length(ν)
        p *= Φ(-ν[i] / σ)
    end
    return p
end

"""
    simulate(model::AbstractLBA; n_steps=100, _...)

Returns a matrix containing evidence samples of the LBA decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::AbstractLBA`: a subtype of AbstractLBA

# Keywords 

- `n_steps=100`: number of time steps at which evidence is recorded
"""
function simulate(rng::AbstractRNG, model::AbstractLBA; n_steps = 100, _...)
    (; τ, A, k, ν, σ) = model
    b = A + k
    n = length(ν)
    νs = sample_drift_rates(rng, ν, σ)
    a = rand(Uniform(0, A), n)
    dt = @. (b - a) / νs
    choice, t = select_winner(dt)
    evidence = collect.(range.(a, a + νs * t, length = 100))
    time_steps = range(0, t, length = n_steps)
    return time_steps, hcat(evidence...)
end
