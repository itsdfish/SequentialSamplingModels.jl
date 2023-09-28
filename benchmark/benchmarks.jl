####################################################################################################
#                                       set up
####################################################################################################
using BenchmarkTools
using SequentialSamplingModels

SUITE = BenchmarkGroup()
####################################################################################################
#                                       logpdf
####################################################################################################
dists2D = (DDM,LBA,LNR,RDM)
dists1D = (Wald,WaldMixture)
ns = [10,100]
SUITE[:logpdf] = BenchmarkGroup()

for dist ∈ dists2D
    dist_name = Symbol(dist)
    for n ∈ ns
        SUITE[:logpdf][dist_name,n] = @benchmarkable(
            logpdf($dist(), data),
            evals=10,
            samples=1000,
            setup=(data=rand($dist(), $n))
        )
    end
end

for dist ∈ dists1D
    dist_name = Symbol(dist)
    for n ∈ ns
        SUITE[:logpdf][dist_name,n] = @benchmarkable(
            logpdf.($dist(), data),
            evals=10,
            samples=1000,
            setup=(data=rand($dist(), $n))
        )
    end
end
####################################################################################################
#                                       rand
####################################################################################################
dists = (DDM,LBA,LCA,LNR,RDM,Wald,WaldMixture)
ns = [10,100]
SUITE[:rand] = BenchmarkGroup()

for dist ∈ dists
    dist_name = Symbol(dist)
    for n ∈ ns
        SUITE[:rand][dist_name,n] = @benchmarkable(
            rand($dist(), $n),
            evals=10,
            samples=1000,
        )
    end
end
####################################################################################################
#                                       simulate
####################################################################################################
dists = (DDM,LBA,LCA,RDM,Wald,WaldMixture)
SUITE[:simulate] = BenchmarkGroup()

for dist ∈ dists
    dist_name = Symbol(dist)
    SUITE[:simulate][dist_name] = @benchmarkable(
        simulate($dist()),
        evals=10,
        samples=1000,
    )
end
# use this to test locally
# results = run(SUITE)