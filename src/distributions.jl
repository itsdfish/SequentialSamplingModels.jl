abstract type Mixed <: ValueSupport end 

const MixedMultivariateDistribution = Distribution{Multivariate, Mixed}

length(MixedMultivariateDistribution) = 2

#sampler()

eltype()

rand()

Distributions._rand!()

Distributions._logpdf()

cdf()

pdf()

minimum()

maximum()

insupport()


struct MyType{T<:Real} <: MixedMultivariateDistribution
    n::Int
    x::T
end

logpdf(d::MyType,data::Int) = logpdf(Binomial(d.n, d.x), data)

loglikelihood(d::MyType,data::Int) =  loglikelihood(Binomial(d.n, d.x), data)