"""
    SequentialSamplingModels

See documentation at: 
https://itsdfish.github.io/SequentialSamplingModels.jl/dev/
"""
module SequentialSamplingModels

using ArgCheck
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
export SSMProductDistribution
export stDDM
export ContinuousMultivariateSSM
export Wald

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
export product_distribution
export rand
export simulate
export std
export survivor

include("type_system.jl")
include("ext_functions.jl")
include("product_distribution.jl")
include("utilities.jl")

include("single_choice_models/ex_gaussian.jl")
include("single_choice_models/ShiftedLogNormal.jl")
include("single_choice_models/wald.jl")

include("multi_choice_models/AttentionalDiffusion.jl")
include("multi_choice_models/ClassicMDFT.jl")
include("multi_choice_models/DDM.jl")
include("multi_choice_models/LBA.jl")
include("multi_choice_models/LCA.jl")
include("multi_choice_models/LNR.jl")
include("multi_choice_models/maaDDM.jl")
include("multi_choice_models/MDFT.jl")
include("multi_choice_models/MLBA.jl")
include("multi_choice_models/poisson_race.jl")
include("multi_choice_models/RDM.jl")
include("multi_choice_models/stDDM.jl")

include("alternative_geometries/CircularDDM.jl")

end
