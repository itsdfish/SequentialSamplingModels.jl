"""
    SequentialSamplingModels

See documentation at: 
https://itsdfish.github.io/SequentialSamplingModels.jl/dev/
"""
module SequentialSamplingModels
    using Distributions
    using DynamicPPL
    using FunctionZeros
    using PrettyTables
    using Random
    using SpecialFunctions

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
    import DynamicPPL: reconstruct
    import DynamicPPL: vectorize
    import StatsAPI: params

    export AbstractaDDM
    export AbstractCDDM
    export AbstractLBA
    export AbstractLCA
    export AbstractLNR
    export AbstractRDM 
    export AbstractWald
    export aDDM
    export CDDM
    export DDM
    export RDM
    export LBA 
    export LCA
    export LNR 
    export maaDDM
    export SSM1D
    export SSM2D
    export ContinuousMultivariateSSM
    export Wald
    export WaldMixture 

    export cdf 
    export loglikelihood 
    export logpdf
    export maximum
    export mean
    export minimum
    export n_options
    export params
    export pdf 
    export rand 
    export reconstruct
    export simulate
    export std
    export vectorize

    include("type_system.jl")
    include("utilities.jl")
    include("LNR.jl")
    include("Wald.jl")
    include("wald_mixture.jl")
    include("LBA.jl")
    include("RDM.jl")
    include("AttentionalDiffusion.jl")
    include("maaDDM.jl")
    include("LCA.jl")
    include("DDM.jl")
    include("CircularDDM.jl")
end