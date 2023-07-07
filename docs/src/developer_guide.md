# Style Guide

In most cases, code written in SequentialSamplingModels.jl follows the guidelines specified in the [blue style](https://github.com/invenia/BlueStyle) guide for Julia. Please use the blue style guide or existing code as a guide, and deviate from the guides only when there is a compelling reason to do so.

# Documentation

## Docstrings

Provide docstrings for methods and types which are part of the API. For example, the doc strings for each model should adhere to the following format:

````markdown
    LNR{T<:Real} <: SSM2D

A lognormal race model object. 

# Parameters 

- `μ`: a vector of means in log-space
- `σ`: a standard deviation parameter in log-space
- `ϕ`: a encoding-response offset

# Constructors

    LNR(μ, σ, ϕ)

    LNR(;μ, σ, ϕ)

# Example

```julia
using SequentialSamplingModels
dist = LNR(μ=[-2,-3], σ=1.0, ϕ=.3)
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
```
# References

Rouder, J. N., Province, J. M., Morey, R. D., Gomez, P., & Heathcote, A. (2015). 
The lognormal race: A cognitive-process model of choice and latency with desirable 
psychometric properties. Psychometrika, 80(2), 491-513.
````

For the benefit of other developers, err on the side of providing doc strings for internal methods. The doc strings should provide the function signature, a high level explanation of the function, and a description of arguments and keywords. Please include references as appropriate. 

## Model Example

Provide a detailed model walk through for the online documentation under the section `Models`. The walk through should include a description of the model, an explanation of the model parameters, and a demonstration showing the pdf overlayed on the histogram (if applicable). Please use existing model examples as a template. 

# API

Only export (make public) types and methods that are intended for users. Other methods are implementational details for interal use. 

# Unit tests

Provide unit tests for most (if not all) methods. When possible, programatically test a method over a wide range of inputs. If you find a bug, write a unit test for the bug to prevent regressions. When possible, compare methods to those defined in established and trusted packages in other languages.  

