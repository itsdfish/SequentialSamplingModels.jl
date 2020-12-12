module SequentialSamplingModels
    using Distributions, Parameters
    import Distributions: pdf, logpdf, rand, loglikelihood, mean, std
    export Wald, WaldMixture, LNR, LBA
    export pdf, logpdf, rand, loglikelihood, mean, std

    include("LogNormalRace.jl")
    include("Wald.jl")
    include("LBA.jl")
end
