using SafeTestsets

files = filter(f -> f ≠ "runtests.jl" && f ≠ "turing.jl", readdir())

include.(files)