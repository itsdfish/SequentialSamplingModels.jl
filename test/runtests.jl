using SafeTestsets

files = filter(f -> f ≠ "runtests.jl", readdir())

include.(files)
