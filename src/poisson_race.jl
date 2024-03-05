"""
    PoissonRace{T<:Real} <: AbstractPoissonRace

# Parameters 

- `ν`: gamma scale parameter
- `α`: threshold
- `τ`: a encoding-response offset

# Constructors

    PoissonRace(ν, α, τ)

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
struct PoissonRace{T<:Real} <: AbstractPoissonRace
    ν::Vector{T}
    α::Vector{Int}
    τ::T
end

function PoissonRace(ν, α, τ)
    _, τ = promote(ν[1], τ)
    ν = convert(Vector{typeof(τ)}, ν)
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

function logpdf(d::AbstractPoissonRace, r::Int, t::Float64)
    (; ν, α, τ) = d
    LL = 0.0
    for i ∈ 1:length(ν)
        if i == r
            LL += logpdf(Gamma(α[i], ν[i]), t - τ)
        else
            LL += logccdf(Gamma(α[i], ν[i]), t - τ)
        end
    end
    return LL
end

function pdf(d::AbstractPoissonRace, r::Int, t::Float64)
    (; ν, α, τ) = d
    density = 1.0
    for i ∈ 1:length(ν)
        if i == r
            density *= pdf(Gamma(α[i], ν[i]), t - τ)
        else
            density *= (1 - cdf(Gamma(α[i], ν[i]), t - τ))
        end
    end
    return density
end

"""
    simulate(model::AbstractPoissonRace; _...)

Returns a matrix containing evidence samples of the LBA decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::AbstractLBA`: a subtype of AbstractLBA

# Keywords 

- `n_steps=100`: number of time steps at which evidence is recorded
"""
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
    return time_steps, reduce(vcat, transpose.(evidence))
end
