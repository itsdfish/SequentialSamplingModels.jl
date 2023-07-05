"""
    SequentialSamplingModels

See documentation at: 
https://itsdfish.github.io/SequentialSamplingModels.jl/dev/
"""
module SequentialSamplingModels
    using Distributions
    using DynamicPPL
    using PrettyTables
    using Random

    import Base: length
    import Distributions: AbstractRNG
    import Distributions: sampler
    import Distributions: cdf
    import Distributions: insupport
    import Distributions: loglikelihood
    import Distributions: logccdf
    import Distributions: logpdf
    import Distributions: maximum
    import Distributions: mean
    import Distributions: minimum
    import Distributions: pdf
    import Distributions: rand
    import Distributions: std
    import DynamicPPL: vectorize
    import StatsAPI: params

    export AbstractaDDM
    export aDDM
    export DDM
    export DiffusionRace
    export LBA 
    export LCA
    export LNR 
    export maaDDM
    export MixedMultivariateDistribution
    export SSM1D
    export SSM2D
    export Wald
    export WaldMixture 

    export cdf 
    export loglikelihood 
    export logpdf
    export maximum
    export mean
    export minimum
    export params
    export pdf 
    export rand 
    export std
    export vectorize

    include("utilities.jl")
    include("LogNormalRace.jl")
    include("Wald.jl")
    include("wald_mixture.jl")
    include("LBA.jl")
    include("DiffusionRace.jl")
    include("AttentionalDiffusion.jl")
    include("maaDDM.jl")
    # include("KDE.jl")
    include("LCA.jl")
    include("DDM.jl")
    include("RatcliffDDM.jl")
end