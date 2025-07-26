# Contributing

If you are interested in contributing, please open an issue and propose changes or additions you think would be beneficial. After discussion and approval, create a fork of the repository, and submit the changes via a pull request. Please review the guidelines described below.  

# Style Guide

The code written in SequentialSamplingModels.jl follows the guidlines presented in the .[JuliaFormatter.toml](https://github.com/itsdfish/SequentialSamplingModels.jl/blob/master/.JuliaFormatter.toml) file, which is inspired by [blue style](https://github.com/JuliaDiff/BlueStyle). You may run the formatter locally by loading SequentialSamplingModels into your session and running ` JuliaFormatter.format(SequentialSamplingModels)`. All PRs undergo a formatting check, which will provide suggestions if you forget to run the formatter locally. 

# API

In this section, we provide a high-level overview of the API. We recommend reviewing the code for the [Lognormal Race Model](https://github.com/itsdfish/SequentialSamplingModels.jl/blob/master/src/multi_choice_models/LNR.jl) for a simple example illustrating the implementation of a model. 

## Types

The type system current consists of the following abstract model types:

- `SSM1D`: abstract SSM for univariate reaction time distributions
- `SSM2D`: abstract SSM for joint choice, reaction time distributions
- `ContinuousMultivariateSSM`: abstract SSM for continuous multivariate distributions

Most models can be implimented as sub-types of one of the three abstract types above. When adding a new model, create a new abstract type so users can develop alternative implementations as needed. 

## Methods 

The following methods are required for each new model. Defining new methods for other functions in the API (e.g., `length`) is typically not required. 

- `increment!`
- `logpdf`
- `pdf`
- `plot_model`
- `rand`
- `simulate`

Only export (make public) types and methods that are intended for users. Other methods are implementational details for interal use. 

# Documentation

Each model should adhere to the following guidelines. Include additional details as necessary.

## Docstrings

Provide docstrings for methods and types which are part of the API. For example, the doc strings for each model should adhere to the following format:

````julia
  LNR{T<:Real} <: AbstractLNR

  Parameters
  ≡≡≡≡≡≡≡≡≡≡

    •  ν: a vector of means in log-space

    •  σ: a vector of standard deviation parameter in log-space

    •  τ: a encoding-response offset

  Constructors
  ≡≡≡≡≡≡≡≡≡≡≡≡

  Two constructors are defined below. The first constructor uses positional arguments, and is therefore order
  dependent:

  LNR(ν, σ, τ)

  The second constructor uses keywords with default values, and is not order dependent:

  LNR(; ν = [-1, -2], σ = fill(1.0, length(ν)), τ = 0.20)

  Example
  ≡≡≡≡≡≡≡

  using SequentialSamplingModels
  dist = LNR(ν = [-2,-3], σ = [1.0,1.0], τ = .3)
  choice,rt = rand(dist, 10)
  like = pdf.(dist, choice, rt)
  loglike = logpdf.(dist, choice, rt)

  References
  ≡≡≡≡≡≡≡≡≡≡

  Rouder, J. N., Province, J. M., Morey, R. D., Gomez, P., & Heathcote, A. (2015). The lognormal race: A
  cognitive-process model of choice and latency with desirable psychometric properties. Psychometrika, 80(2), 491-513.
````

For the benefit of other developers, err on the side of providing doc strings for internal methods. The doc strings should provide the function signature, a high level explanation of the function, and a description of arguments and keywords. Please include references as appropriate. 

## Model Example

Provide a detailed model walk through for the online documentation under the section `Models`. The walk through should include a description of the model, an explanation of the model parameters, and a demonstration showing the pdf overlayed on the histogram (if applicable). Please use existing model examples as a template. 


# Unit Tests

Provide unit tests for most (if not all) methods. When possible, programatically test a method over a wide range of inputs. If you find a bug, write a unit test for the bug to prevent regressions. When possible, compare methods to those defined in established and trusted packages in other languages.  

# Parameter Naming Conventions
To ensure consistency across models, please use the following variable names:

1. use `ν` for drift rates
2. use `α` for decision boundaries
3. use `Δt` for a discrete time step
4. use `σ` for within-trial noise of drift rate  
5. use `τ` for non-decision time
6. use `z` for evidence starting point
7. use `η` for across-trial noise of drift rate

# Other Conventions

Use variable names that are descriptive unless there is a strong mathematical convention for a particular variable name. When appropriate, use verbs to describe functions. For example, use `summarize(model)` instead of `summary(model)`. Use lower case for variables, and capitalize the first letter of package names, types, and constructors. Use underscore to separate words. For example, name a file `developer_guide.md` instead of `developerguide.md`.

