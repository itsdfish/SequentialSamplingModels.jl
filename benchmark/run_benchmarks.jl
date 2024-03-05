cd(@__DIR__)
####################################################################################################
#                                      load packages
####################################################################################################
using BenchmarkPlots
using DataFrames
using PkgBenchmark
using SequentialSamplingModels
using StatsPlots
####################################################################################################
#                                       benchmark
####################################################################################################
baselin_id = "0e32c56acea48fd722cd8176cd7e3d1d21d8b93c"
baseline = benchmarkpkg(SequentialSamplingModels, baselin_id)

target_id = "2db1b3752ec8ffa88b8bf69c480cc190e4d395f2"
target = benchmarkpkg(SequentialSamplingModels, target_id)

comparison = judge(target, baseline)
export_markdown("judgement_test.md", comparison)
####################################################################################################
#                                       plots
####################################################################################################

# plot(local_benchmark[:simulate])

# df = DataFrame(median(local_benchmark[:logpdf]))
# transform!(df,
#           :first => ByRow(identity) => [:model,:n],
#           :second => (t -> time.(t)) => "time")
# @df df groupedbar(:time, group=(:model,:n))
