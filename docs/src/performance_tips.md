# Performance Tips

## General Tips

In Julia, high performance can be achieved by following a small set of principles, such as avoiding global variables, avoding heterogenous containers, and placing performance critical code in a function. The same basic principles apply when using `SequentialSamplingModels.jl`. See the [Julia documentation](https://docs.julialang.org/en/v1/manual/performance-tips/) for more details.

## Turing

Turing provides three general recommendations for developing performant code:

1. Ensure types are inferable using principles defined in the [Julia documentation](https://docs.julialang.org/en/v1/manual/performance-tips/)
2. Use Multivariate distributions in place of Univariate distributions when applicable. 
3. Use forward mode automatic differentiation when your model has a small number of parameters (i.e., 5-10), and use reverse mode automatic differentiation for larger models. 

See the [Turing documentation](https://turinglang.org/docs/tutorials/docs-13-using-turing-performance-tips/) for more details. Note that the Turing ecosystem provides a benchmarking package, [TuringBenchmarking.jl](https://turinglang.org/TuringBenchmarking.jl/dev/) to aid in the selection of an automatic differentiation backend. 