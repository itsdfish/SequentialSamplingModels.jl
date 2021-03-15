"""
# SequentialSamplingModels.jl
A collection of sequential sampling models based on the Distributions.jl API.

## Currently Supported models

- `LBA`: Linear Ballistic Accumulator
- `LNR`: Lognormal Race Model
- `Wald`: a shifted Wald represented a single boundary diffusion process
- `WaldMixture`: a shifted Wald represented a single boundary diffusion process with across-trial 
    variability in the drift rate
"""
module SequentialSamplingModels
    using Distributions, Parameters
    import Distributions: pdf, logpdf, rand, loglikelihood, mean, std, cdf
    import Distributions: logccdf
    export Wald, WaldMixture, LNR, LBA, DiffusionRace
    export pdf, cdf, logpdf, rand, loglikelihood, mean, std

    include("LogNormalRace.jl")
    include("Wald.jl")
    include("LBA.jl")
    include("DiffusionRace.jl")
end
