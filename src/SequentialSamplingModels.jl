"""
    SequentialSamplingModels

See documentation at: 
https://itsdfish.github.io/SequentialSamplingModels.jl/dev/
"""
module SequentialSamplingModels
    using Distributions
    using ConcreteStructs
    using PrettyTables

    import Distributions: cdf
    import Distributions: loglikelihood
    import Distributions: logccdf
    import Distributions: logpdf
    import Distributions: maximum
    import Distributions: mean
    import Distributions: minimum
    import Distributions: pdf
    import Distributions: rand
    import Distributions: std
    import StatsAPI: params

    export SequentialSamplingModel
    export AbstractaDDM
    export aDDM
    export DDM
    export DiffusionRace
    export LBA 
    export LCA
    export LNR 
    export maaDDM
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

    include("utilities.jl")
    include("LogNormalRace.jl")
    include("Wald.jl")
    include("LBA.jl")
    include("DiffusionRace.jl")
    include("AttentionalDiffusion.jl")
    include("KDE.jl")
    include("LCA.jl")
    include("DDM.jl")
end