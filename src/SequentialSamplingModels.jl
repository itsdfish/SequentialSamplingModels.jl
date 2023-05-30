"""
# SequentialSamplingModels.jl

A collection of sequential sampling models based on the Distributions.jl API.

## Currently Supported models

- `LBA`: Linear Ballistic Accumulator
- `LNR`: Lognormal Race Model
- `Wald`: a shifted Wald represented a single boundary diffusion process
- `WaldMixture`: a shifted Wald represented a single boundary diffusion process with across-trial 
    variability in the drift rate
- `aDDM`: a drift diffusion model in which the accumulation process is determined by the 
utility of a visually attended option
"""
module SequentialSamplingModels
    using Distributions
    using ConcreteStructs
    using PrettyTables

    import Distributions: pdf
    import Distributions: logpdf
    import Distributions: rand
    import Distributions: loglikelihood
    import Distributions: mean
    import Distributions: std
    import Distributions: cdf
    import Distributions: logccdf

    export SequentialSamplingModel
    export Wald
    export WaldMixture 
    export LNR 
    export LBA 
    export DiffusionRace
    export AbstractaDDM
    export aDDM
    export maaDDM
    export LCA

    export pdf 
    export cdf 
    export logpdf 
    export rand 
    export loglikelihood 
    export mean 
    export std

    include("utilities.jl")
    include("LogNormalRace.jl")
    include("Wald.jl")
    include("LBA.jl")
    include("DiffusionRace.jl")
    include("AttentionalDiffusion.jl")
    include("KDE.jl")
    include("LCA.jl")
end