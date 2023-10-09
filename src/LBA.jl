"""
    LBA{T<:Real} <: AbstractLBA

A model object for the linear ballistic accumulator.

# Parameters

- `ν`: a vector of drift rates
- `A`: max start point
- `k`: A + k = b, where b is the decision threshold
- `σ`: a vector of drift rate standard deviation
- `τ`: a encoding-response offset

# Constructors 

    LBA(ν, A, k, τ, σ)
    
    LBA(;τ=.3, A=.8, k=.5, ν=[2.0,1.75], σ=[1.0,1.0])

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
mutable struct LBA{T<:Real} <: AbstractLBA
    ν::Vector{T}
    A::T
    k::T
    τ::T
    σ::Vector{T}
end

function LBA(ν, A, k, τ, σ)
    _, A, k, τ, _ = promote(ν[1], A, k, τ, σ[1])
    ν = convert(Vector{typeof(k)}, ν)
    σ = convert(Vector{typeof(k)}, σ)
    return LBA(ν, A, k, τ, σ)
end

function params(d::LBA)
    return (d.ν,d.A,d.k,d.τ,d.σ)    
end

LBA(;τ=.3, A=.8, k=.5, ν=[2.0,1.75], σ=fill(1.0, length(ν))) = LBA(ν, A, k, τ, σ)

function select_winner(dt)
    if any(x -> x > 0, dt)
        mi,mv = 0,Inf
        for (i,t) in enumerate(dt)
            if (t > 0) && (t < mv)
                mi = i
                mv = t
            end
        end
    else
        return 1,-1.0
    end
    return mi,mv
end

sample_drift_rates(ν, σ) = sample_drift_rates(Random.default_rng(), ν, σ)

function sample_drift_rates(rng::AbstractRNG, ν, σ)
    negative = true
    v = similar(ν)
    n_options = length(ν)
    while negative
        v = [rand(rng, Normal(ν[i], σ[i])) for i ∈ 1:n_options]
        negative = any(x -> x > 0, v) ? false : true
    end
    return v
end

function rand(rng::AbstractRNG, d::AbstractLBA)
    (;τ,A,k,ν,σ) = d
    b = A + k
    N = length(ν)
    v = sample_drift_rates(rng, ν, σ)
    a = rand(rng, Uniform(0, A), N)
    dt = @. (b - a) / v
    choice,mn = select_winner(dt)
    rt = τ .+ mn
    return (;choice,rt)
end

function logpdf(d::AbstractLBA, c, rt)
    (;τ,A,k,ν,σ) = d
    b = A + k; den = 0.0
    rt < τ ? (return -Inf) : nothing
    for i ∈ 1:length(ν)
        if c == i
            den += log_dens(d, ν[i], σ[i], rt)
        else
            den += log(max(0.0, 1 - cummulative(d, ν[i], σ[i], rt)))
        end
    end
    pneg = pnegative(d)
    den = den - log(1 - pneg)
    return max(den, -1000.0)
end

function pdf(d::AbstractLBA, c, rt)
    (;τ,A,k,ν,σ) = d
    b = A + k; den = 1.0
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

function dens(d::AbstractLBA, v, σ, rt)
    (;τ,A,k) = d
    dt = rt - τ; b = A + k
    n1 = (b - A - dt * v) / (dt * σ)
    n2 = (b - dt * v) / (dt * σ)
    𝒩 = Normal(0, 1)
    dens = (1 / A) * (-v * cdf(𝒩, n1) + σ * pdf(𝒩, n1) +
        v * cdf(𝒩, n2) - σ * pdf(𝒩, n2))
    return max(dens, 0.0)
end

function log_dens(d::AbstractLBA, v, σ, rt)
    (;τ,A,k) = d
    dt = rt - τ; b = A + k
    n1 = (b - A - dt * v) / (dt * σ)
    n2 = (b - dt * v) / (dt * σ)
    𝒩 = Normal(0, 1)
    dens = -log(A) + log(max(0.0, -v * cdf(𝒩, n1) + σ * pdf(𝒩, n1) +
        v * cdf(𝒩, n2) - σ * pdf(𝒩, n2)))
    return dens
end

function cummulative(d::AbstractLBA, v, σ, rt)
    (;τ,A,k) = d
    dt = rt - τ; b = A + k
    n1 = (b - A - dt * v) / (dt * σ)
    n2 = (b - dt * v) / (dt * σ)
    𝒩 = Normal(0, 1)
    cm = 1 + ((b - A -dt * v) / A) * cdf(𝒩, n1) -
        ((b - dt * v) / A) * cdf(𝒩, n2) + ((dt * σ) / A) * pdf(𝒩, n1) -
        ((dt * σ) / A) * pdf(𝒩, n2)
    return cm
end

function pnegative(d::AbstractLBA)
    (;ν,σ) = d
    p = 1.0
    𝒩 = Normal(0, 1)
    for i ∈ 1:length(ν)
        p *= cdf(𝒩, -ν[i] / σ[i])
    end
    return p
end

"""
    simulate(model::AbstractLBA; n_steps=100)

Returns a matrix containing evidence samples of the LBA decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::AbstractLBA`: a subtype of AbstractLBA

# Keywords 

- `n_steps=100`: number of time steps at which evidence is recorded
"""
function simulate(model::AbstractLBA; n_steps=100)
    (;τ,A,k,ν,σ) = model
    b = A + k
    n = length(ν)
    νs = sample_drift_rates(ν, σ)
    a = rand(Uniform(0, A), n)
    dt = @. (b - a) / νs
    choice,t = select_winner(dt)
    evidence = collect.(range.(a, a + νs * t, length=100))
    time_steps = range(0, t, length=n_steps)
    return time_steps,hcat(evidence...)
end