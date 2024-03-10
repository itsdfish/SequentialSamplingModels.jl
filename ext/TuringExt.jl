module TuringExt

import DynamicPPL: reconstruct
import DynamicPPL: vectorize
using SequentialSamplingModels
import SequentialSamplingModels: predict_distribution
import SequentialSamplingModels: predict_density
using Turing: @model
using Turing: @submodel

"""
    predict_distribution(dist, args...; model, func, n_samples, kwargs...)

Generates a predictive distribution for a statistic defined by `func`.

# Arguments 

- `dist::Distribution`: a distribution type which accepts parameters as keyword arguments 
- `args...` optional positional arguments passed to function `func`

# Keywords

- `model`: a Turing model which returns a `NamedTuple` of parameters 
- `func`: a function which computes a statistic of simulated data. The function signature is `func(data, args...; kwargs...)`
- `n_samples`: the number of observations to sample from `dist`
- `kwargs...`: optional keyword arguments passed to `func`
"""
@model function predict_distribution(dist, args...; model, func, n_samples, kwargs...)
    @submodel parms = model
    sim_data = rand(dist(; parms...), n_samples)
    return func(sim_data, args...; kwargs...)
end

"""
    predict_density(dist; model, t_range=range(.2, .8, length=100))

Generates a predictive distribution of the probability density function.

# Arguments 

- `dist::Distribution`: a distribution type which accepts parameters as keyword arguments 

# Keywords

- `model`: a Turing model which returns a `NamedTuple` of parameters 
- `t_range=range(.2, .8, length=100)`: time points at which the density is computed
"""
@model function predict_density(dist; model, t_range = range(0.2, 0.8, length = 100))
    @submodel parms = model
    return pdf.(dist(; parms...), t_range)
end

vectorize(d::SSM2D, r::NamedTuple) = [r...]
reconstruct(d::SSM2D, v::NamedTuple) = deepcopy(v)
end
