####################################################################################################
#                                       set up
####################################################################################################
using BenchmarkTools
using SequentialSamplingModels
using DataFrames

SUITE = BenchmarkGroup()
####################################################################################################
#                                       logpdf
####################################################################################################
dists2D = (DDM, LBA, LNR, RDM)
dists1D = (Wald,)
ns = [10, 100]
SUITE[:logpdf] = BenchmarkGroup()

for dist ∈ dists2D
    dist_name = Symbol(dist)
    for n ∈ ns
        SUITE[:logpdf][dist_name, n] = @benchmarkable(
            logpdf($dist(), data),
            evals = 10,
            samples = 1000,
            setup = (data = rand($dist(), $n))
        )
    end
end

for dist ∈ dists1D
    dist_name = Symbol(dist)
    for n ∈ ns
        SUITE[:logpdf][dist_name, n] = @benchmarkable(
            logpdf.($dist(), data),
            evals = 10,
            samples = 1000,
            setup = (data = rand($dist(), $n))
        )
    end
end
####################################################################################################
#                                       rand
####################################################################################################
dists = (DDM, LBA, LCA, LNR, RDM, Wald)
ns = [10, 100]
SUITE[:rand] = BenchmarkGroup()

for dist ∈ dists
    dist_name = Symbol(dist)
    for n ∈ ns
        SUITE[:rand][dist_name, n] =
            @benchmarkable(rand($dist(), $n), evals = 10, samples = 1000,)
    end
end
####################################################################################################
#                                       simulate
####################################################################################################
dists = (DDM, LBA, LCA, RDM, Wald)
SUITE[:simulate] = BenchmarkGroup()

for dist ∈ dists
    dist_name = Symbol(dist)
    SUITE[:simulate][dist_name] =
        @benchmarkable(simulate($dist()), evals = 10, samples = 1000,)
end

parms = (
    σ = 0.1,
    α = 0.50,
    τ = 0.0,
    γ = 1.0,
    ϕ1 = 0.01,
    ϕ2 = 0.1,
    β = 10,
    κ = [5, 5]
)

mdft = MDFT(; n_alternatives = 3, parms...)

M = [
    1.0 3.0
    3.0 1.0
    0.9 3.1
]

SUITE[:simulate][:mdft] = @benchmarkable(simulate(mdft, M), evals = 10, samples = 1000,)
