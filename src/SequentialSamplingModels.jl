"""
    SequentialSamplingModels

See documentation at: 
https://itsdfish.github.io/SequentialSamplingModels.jl/dev/
"""
module SequentialSamplingModels

using Distributions
using FunctionZeros
using PrettyTables
using Random
using SpecialFunctions

using Distributions: ProductDistribution
using HCubature: hcubature
using StatsBase: Weights

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
import Distributions: rand!
import Distributions: std
import StatsAPI: params
import StatsBase: cor2cov

export AbstractDDM
export AbstractaDDM
export AbstractCDDM
export AbstractLBA
export AbstractLCA
export AbstractLNR
export AbstractMLBA
export AbstractMDFT
export AbstractPoissonRace
export AbstractRDM
export AbstractShiftedLogNormal
export AbstractstDDM
export AbstractWald
export aDDM
export ClassicMDFT
export CDDM
export DDM
export ExGaussian
export RDM
export LBA
export LCA
export LNR
export maaDDM
export MLBA
export MDFT
export PoissonRace
export ShiftedLogNormal
export SSM1D
export SSM2D
export stDDM
export ContinuousMultivariateSSM
export Wald
export WaldMixture

export cdf
export compute_choice_probs
export compute_quantiles
export loglikelihood
export logpdf
export maximum
export mean
export minimum
export n_options
export params
export pdf
export plot_choices
export plot_choices!
export plot_model
export plot_model!
export plot_quantiles
export plot_quantiles!
export predict_distribution
export rand
export simulate
export std
export survivor

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
include("ext_functions.jl")
include("ex_gaussian.jl")
include("poisson_race.jl")
include("stDDM.jl")
include("MDFT.jl")
include("ClassicMDFT.jl")
include("MLBA.jl")
include("ShiftedLogNormal.jl")
include("product_distribution.jl")
end
