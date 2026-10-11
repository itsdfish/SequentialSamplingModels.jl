# Performance Tips

## General Tips

In Julia, high performance can be achieved by following a small set of principles, such as avoiding global variables, avoding heterogenous containers, and placing performance critical code in a function. The same basic principles apply when using `SequentialSamplingModels.jl`. See the [Julia documentation](https://docs.julialang.org/en/v1/manual/performance-tips/) for more details.

## Turing

Turing provides three general recommendations for developing performant code:

1. Ensure types are inferable using principles defined in the [Julia documentation](https://docs.julialang.org/en/v1/manual/performance-tips/)
2. Use Multivariate distributions in place of Univariate distributions when applicable. 
3. Use forward mode automatic differentiation when your model has a small number of parameters (i.e., 5-10), and use reverse mode automatic differentiation for larger models. 

See the [Turing documentation](https://turinglang.org/docs/tutorials/docs-13-using-turing-performance-tips/) for more details. Note that the Turing ecosystem provides a benchmarking package, [TuringBenchmarking.jl](https://turinglang.org/TuringBenchmarking.jl/dev/) to aid in the selection of an automatic differentiation backend.

## Automatic Differentiation Backends

Do not use a compiled ReverseDiff tape (`AutoReverseDiff(; compile = true)`) with the `DDM`.

A compiled ReverseDiff tape reuses the calculation steps recorded on its first run. The DDM changes its calculation and number of series terms as parameters change, so reusing those steps can give incorrect gradients without an error. This also affects other models whose series length depends on their parameters.

The following timings measure one gradient evaluation for a hierarchical DDM log likelihood:

| parameters | observations | ForwardDiff | ReverseDiff (compiled) | Mooncake |
|---|---|---|---|---|
| 53 | 1,000 | 3.8 ms | 9.8 ms | **1.2 ms** |
| 153 | 3,000 | 38.2 ms | 34.7 ms | **3.8 ms** |
| 403 | 8,000 | 290.1 ms | 98.4 ms | **11.3 ms** |

Mooncake was fastest in these benchmarks and supports calculations that change with the parameters. Use `AutoMooncake` for larger models and `AutoForwardDiff` for small ones. Uncompiled ReverseDiff (`AutoReverseDiff(; compile = false)`) also works, but is slower.