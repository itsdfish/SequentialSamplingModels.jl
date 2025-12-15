"""
    PoissonRace{T<:Real} <: AbstractPoissonRace

# Parameters 

- `ν::T`: gamma scale parameter. ν ∈ ℝ⁺.
- `α::T`: threshold. α ∈ ℝ⁺
- `τ::T`: a encoding-response offset. τ ∈ [0, min_rt].

# Constructors

Two constructors are defined below. The first constructor uses positional arguments, and is therefore order dependent:

    PoissonRace(ν, α, τ)

The second constructor uses keywords with default values, and is not order dependent: 

    PoissonRace(;ν=[.05,.06], α=[5,5], τ=.3)

# Example

```julia
using SequentialSamplingModels
dist = PoissonRace(ν=[.05,.06], α=[5,5], τ=.3)
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
```
# References

LaBerge, D. A. (1962). A recruitment model of simple behavior. Psychometrika, 27, 375-395.
"""
struct PoissonRace{T <: Real} <: AbstractPoissonRace
    ν::Vector{T}
    α::Vector{Int}
    τ::T
    function PoissonRace(ν::Vector{T}, α::Vector{Int}, τ::T) where {T <: Real}
        @argcheck length(ν) == length(α)
        @argcheck all(α .≥ 0)
        @argcheck τ ≥ 0
        return new{T}(ν, α, τ)
    end
end

function PoissonRace(ν, α, τ::T) where {T}
    _, τ = promote(ν[1], τ)
    ν = convert(Vector{T}, ν)
    return PoissonRace(ν, α, τ)
end

function params(d::AbstractPoissonRace)
    return (d.ν, d.α, d.τ)
end

PoissonRace(; ν = [0.05, 0.06], α = [5, 5], τ = 0.30) = PoissonRace(ν, α, τ)

function rand(rng::AbstractRNG, dist::AbstractPoissonRace)
    (; ν, α, τ) = dist
    x = @. rand(rng, Gamma(α, ν)) + τ
    rt, choice = findmin(x)
    return (; choice, rt)
end

function logpdf(d::AbstractPoissonRace, r::Int, rt::Float64)
    (; ν, α, τ) = d
    @argcheck τ ≤ rt
    LL = 0.0
    for i ∈ 1:length(ν)
        if i == r
            LL += logpdf(Gamma(α[i], ν[i]), rt - τ)
        else
            LL += logccdf(Gamma(α[i], ν[i]), rt - τ)
        end
    end
    return LL
end

function pdf(d::AbstractPoissonRace, r::Int, rt::Float64)
    (; ν, α, τ) = d
    @argcheck τ ≤ rt
    density = 1.0
    for i ∈ 1:length(ν)
        if i == r
            density *= pdf(Gamma(α[i], ν[i]), rt - τ)
        else
            density *= (1 - cdf(Gamma(α[i], ν[i]), rt - τ))
        end
    end
    return density
end

function simulate(model::AbstractPoissonRace; Δt = 0.001)
    (; ν, α, τ) = model
    n = n_options(model)
    counts = fill(0, n)
    evidence = [fill(0.0, n)]
    time_steps = [0.0]
    t = 0.0
    count_times = @. rand(Exponential(ν))

    while all(counts .< α)
        t += Δt
        for i ∈ 1:n
            if t > count_times[i]
                count_times[i] += rand(Exponential(ν[i]))
                counts[i] += 1
            end
        end
        push!(evidence, counts)
        push!(time_steps, t)
    end
    return time_steps, stack(evidence, dims = 1)
end
