function rand(
    rng::AbstractRNG,
    s::Sampleable{T, R}
) where {T <: Matrixvariate, R <: SequentialSamplingModels.Mixed}
    n = size(s, 2)
    data = (; choice = fill(0, n), rt = fill(0.0, n))
    return rand!(rng, s, data)
end

function rand(
    rng::AbstractRNG,
    s::Sampleable{T, R},
    dims::Dims
) where {T <: Matrixvariate, R <: SequentialSamplingModels.Mixed}
    n = size(s, 2)
    ax = map(Base.OneTo, dims)
    data = [(; choice = fill(0, n), rt = fill(0.0, n)) for _ in Iterators.product(ax...)]
    return [rand!(rng, s, d) for d ∈ data]
end

function rand!(
    rng::AbstractRNG,
    s::Sampleable{T, R},
    data::NamedTuple
) where {T <: Matrixvariate, R <: SequentialSamplingModels.Mixed}
    for i ∈ 1:size(s, 2)
        data.choice[i], data.rt[i] = rand(rng, s.dists[i])
    end
    return data
end

function logpdf(d::ProductDistribution, data_array::Array{<:NamedTuple, N}) where {N}
    return [logpdf(d, data) for data ∈ data_array]
end

function logpdf(d::ProductDistribution, data::NamedTuple)
    LL = 0.0
    for i ∈ 1:length(d.dists)
        LL += logpdf(d.dists[i], data.choice[i], data.rt[i])
    end
    return LL
end
