"""
* `μ`: a vector of means in log-space
* `σ`: a standard deviation parameter in log-space
* `ϕ`: a encoding-response offset
"""
struct LNR{T1,T2,T3} <: ContinuousUnivariateDistribution
    μ::T1
    σ::T2
    ϕ::T3
end

Broadcast.broadcastable(x::LNR) = Ref(x)

LNR(;μ, σ, ϕ) = LNR(μ, σ, ϕ)

function rand(dist::LNR)
    @unpack μ,σ,ϕ = dist
    x = @. rand(LogNormal(μ, σ)) + ϕ
    rt,resp = findmin(x)
    return resp,rt
end

rand(dist::LNR, N::Int) = [rand(dist) for i in 1:N]

function logpdf(d::T, r::Int, t::Float64) where {T<:LNR}
    @unpack μ,σ,ϕ = d
    LL = 0.0
    for (i,m) in enumerate(μ)
        if i == r
            LL += logpdf(LogNormal(m, σ), t - ϕ)
        else
            LL += logccdf(LogNormal(m, σ), t - ϕ)
        end
    end
    return LL
end

function logpdf(d::LNR{T1,T2,Vector{T3}}, r::Int, t::Float64) where {T1,T2,T3}
    @unpack μ,σ,ϕ = d
    LL = 0.0
    for (i,m) in enumerate(μ)
        if i == r
            LL += logpdf(LogNormal(m, σ), t - ϕ[i])
        else
            LL += logccdf(LogNormal(m, σ), t - ϕ[i])
        end
    end
    return LL
end

logpdf(d::LNR, data::Tuple) = logpdf(d, data...)

pdf(d::LNR, data::Tuple) = pdf(d, data...)

function pdf(d::T, r::Int, t::Float64) where {T<:LNR}
    @unpack μ,σ,ϕ = d
    density = 1.0
    for (i,m) in enumerate(μ)
        if i == r
            density *= pdf(LogNormal(m, σ), t-ϕ)
        else
            density *= (1-cdf(LogNormal(m, σ), t-ϕ))
        end
    end
    return density
end
