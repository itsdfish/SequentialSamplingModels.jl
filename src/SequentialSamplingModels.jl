"""
# SequentialSamplingModels.jl

A collection of sequential sampling models based on the Distributions.jl API.

## Currently Supported models

- `LBA`: Linear Ballistic Accumulator
- `LNR`: Lognormal Race Model
- `Wald`: a shifted Wald represented a single boundary diffusion process
- `WaldMixture`: a shifted Wald represented a single boundary diffusion process with across-trial 
    variability in the drift rate
- `AttentionalDiffusion`: a drift diffusion model in which the accumulation process is determined by the 
utility of a visually attended option
"""
module SequentialSamplingModels
    using Distributions, Parameters, ConcreteStructs, PrettyTables
    using KernelDensity, Interpolations
    import KernelDensity: kernel_dist
    import Distributions: pdf, logpdf, rand, loglikelihood, mean, std, cdf
    import Distributions: logccdf
    export SequentialSamplingModel, 
        Wald, 
        WaldMixture, 
        LNR, 
        LBA, 
        DiffusionRace, 
        AttentionalDiffusion

    export pdf, 
        cdf, 
        logpdf, 
        rand, 
        loglikelihood, 
        mean, 
        std

    include("utilities.jl")
    include("LogNormalRace.jl")
    include("Wald.jl")
    include("LBA.jl")
    include("DiffusionRace.jl")
    include("AttentionalDiffusion.jl")
    include("KDE.jl")
end
