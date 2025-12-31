"""
    SSMProductDistribution

Wrapper around `ProductDistribution` for sequential sampling models.
This type allows us to define `logpdf` methods for `NamedTuple` data
without type piracy.
"""
struct SSMProductDistribution{D <: ProductDistribution}
    dist::D
end

"""
    product_distribution(dists)

Create a product distribution from a vector of distributions.
Returns an `SSMProductDistribution` for SSM types, or a standard
`ProductDistribution` for other types.
"""
function product_distribution(dists::AbstractVector)
    pd = ProductDistribution(dists)
    # Check if this is an SSM that produces NamedTuple data
    if eltype(dists) <: SSM2D
        return SSMProductDistribution(pd)
    else
        return pd
    end
end

Base.size(s::SSMProductDistribution, dims...) = size(s.dist, dims...)
Base.length(s::SSMProductDistribution) = length(s.dist)

function rand(
    rng::AbstractRNG,
    s::SSMProductDistribution
)
    n = size(s, 2)
    data = (; choice = fill(0, n), rt = fill(0.0, n))
    return rand!(rng, s, data)
end

function rand(
    rng::AbstractRNG,
    s::SSMProductDistribution,
    dims::Dims
)
    n = size(s, 2)
    ax = map(Base.OneTo, dims)
    data = [(; choice = fill(0, n), rt = fill(0.0, n)) for _ in Iterators.product(ax...)]
    return [rand!(rng, s, d) for d ∈ data]
end

function rand!(
    rng::AbstractRNG,
    s::SSMProductDistribution,
    data::NamedTuple
)
    for i ∈ 1:size(s, 2)
        data.choice[i], data.rt[i] = rand(rng, s.dist.dists[i])
    end
    return data
end

function logpdf(d::SSMProductDistribution, data_array::Array{<:NamedTuple, N}) where {N}
    return [logpdf(d, data) for data ∈ data_array]
end

function logpdf(d::SSMProductDistribution, data::NamedTuple)
    LL = 0.0
    for i ∈ 1:length(d.dist.dists)
        LL += logpdf(d.dist.dists[i], data.choice[i], data.rt[i])
    end
    return LL
end
