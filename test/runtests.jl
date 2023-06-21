using SafeTestsets

files = filter(f -> f ≠ "runtests.jl", readdir())

# only use DDM tests for now
files = ["ddm_tests.jl"]


include.(files)