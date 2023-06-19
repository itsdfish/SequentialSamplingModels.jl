
"""
    LBA(;τ, A, k, ν, σ=1.0)

A model object for the linear ballistic accumulator.

# Fields

- `ν`: a vector of drift rates
- `A`: max start point
- `k`: A + k = b, where b is the decision threshold
- `σ=1`: drift rate standard deviation
- `τ`: a encoding-response offset

# Example 

````julia
using SequentialSamplingModels
dist = LBA(ν=[3.0,2.0], A = .8, k = .2, τ = .3) 
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
````

# References

Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice response time: Linear ballistic accumulation. Cognitive psychology, 57(3), 153-178.
"""
mutable struct LBA{T1,T2,T3,T4} <: SequentialSamplingModel
    ν::T1
    A::T2
    k::T3
    τ::T4
    σ::Float64
end

Base.broadcastable(x::LBA) = Ref(x)

function params(d::LBA)
    (d.ν,d.A,d.k,d.τ,d.σ)    
end

loglikelihood(d::LBA, data) = sum(logpdf.(d, data...))

LBA(;τ, A, k, ν, σ=1.0) = LBA(ν, A, k, τ, σ)

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

function sample_drift_rates(rng::AbstractRNG, ν, σ)
    negative = true
    v = similar(ν)
    while negative
        v = [rand(rng, Normal(d, σ)) for d in ν]
        negative = any(x -> x > 0, v) ? false : true
    end
    return v
end

function rand(rng::AbstractRNG, d::LBA)
    (;τ,A,k,ν,σ) = d
    b = A + k
    N = length(ν)
    v = sample_drift_rates(rng, ν, σ)
    a = rand(rng, Uniform(0, A), N)
    dt = @. (b - a) / v
    choice,mn = select_winner(dt)
    rt = τ .+ mn
    return choice,rt
end

function rand(rng::AbstractRNG, d::LBA, N::Int)
    choice = fill(0, N)
    rt = fill(0.0, N)
    for i in 1:N
        choice[i],rt[i] = rand(rng, d)
    end
    return (choice=choice,rt=rt)
end

logpdf(d::LBA, choice, rt) = log(pdf(d, choice, rt))

function logpdf(d::LBA, data::T) where {T<:NamedTuple}
    return sum(logpdf.(d, data...))
end

function logpdf(dist::LBA, data::Array{<:Tuple,1})
    LL = 0.0
    for d in data
        LL += logpdf(dist, d...)
    end
    return LL
end

function pdf(d::LBA, c, rt)
    (;τ,A,k,ν,σ) = d
    b = A + k; den = 1.0
    rt < τ ? (return 1e-10) : nothing
    for (i,v) in enumerate(ν)
        if c == i
            den *= dens(d, v, rt)
        else
            den *= (1 - cummulative(d, v, rt))
        end
    end
    pneg = pnegative(d)
    den = den / (1 - pneg)
    den = max(den, 1e-10)
    isnan(den) ? (return 0.0) : (return den)
end

logpdf(d::LBA, data::Tuple) = logpdf(d, data...)

function dens(d::LBA, v, rt)
    (;τ,A,k,ν,σ) = d
    dt = rt - τ; b = A + k
    n1 = (b - A - dt * v) / (dt * σ)
    n2 = (b - dt * v) / (dt * σ)
    dens = (1 / A) * (-v * cdf(Normal(0, 1), n1) + σ * pdf(Normal(0,1), n1) +
        v * cdf(Normal(0,1), n2) - σ * pdf(Normal(0,1) ,n2))
    return dens
end

function cummulative(d::LBA, v, rt)
    (;τ,A,k,ν,σ) = d
    dt = rt - τ; b = A + k
    n1 = (b - A - dt * v) / (dt * σ)
    n2 = (b - dt * v) / (dt * σ)
    cm = 1 + ((b - A -dt * v) / A) * cdf(Normal(0, 1), n1) -
        ((b - dt * v) / A) * cdf(Normal(0, 1), n2) + ((dt * σ) / A)*pdf(Normal(0, 1), n1) -
        ((dt * σ) / A) * pdf(Normal(0, 1), n2)
    return cm
end

function pnegative(d::LBA)
    (;ν,σ) = d
    p = 1.0
    for v in ν
        p *= cdf(Normal(0, 1), -v / σ)
    end
    return p
end
#add Distribution methods for ContinuousUnivariateDistribution
#Distribution.rand(rng::AbstractRNG, d::LBA) = 