@safetestset "compute_quantiles" begin
    @safetestset "SSM2D" begin
        using SequentialSamplingModels
        using Test

        dist = LBA(ν = [3.0, 3.0], A = 0.8, k = 0.2, τ = 0.3)
        data = rand(dist, 100)
        percentiles = [0.3, 0.5, 0.7]
        qs = compute_quantiles(data; percentiles)
        @test length.(qs) == [3, 3]

        qs = compute_quantiles(data; percentiles, choice_set = [:a, :b])
        @test length(qs) == 2
        @test isempty(qs[1])
        @test isempty(qs[2])
    end

    @safetestset "ContinuousMultivariateSSM" begin
        using SequentialSamplingModels
        using Test

        dist = CDDM(; ν = [1, 0.5], η = [1, 1], σ = 1, α = 1.5, τ = 0.30)
        data = rand(dist, 10)
        percentiles = [0.3, 0.5, 0.7]
        qs = compute_quantiles(data; percentiles)
        @test length.(qs) == [3, 3]
    end

    @safetestset "1D SSM" begin
        using SequentialSamplingModels
        using Test

        dist = Wald(ν = 3.0, α = 0.5, τ = 0.130)
        rts = rand(dist, 10)
        percentiles = [0.3, 0.5, 0.7]
        qs = compute_quantiles(rts; percentiles)
        @test length(qs) == 3
    end
end

@safetestset "compute_choice_probs" begin
    using SequentialSamplingModels
    using Test

    data = (choice = [1, 1, 2], rts = [0.3, 0.5, 0.3])
    choice_probs = compute_choice_probs(data; choice_set = [1, 2])

    @test choice_probs == [2 / 3, 1 / 3]
end

@safetestset "Survivor" begin
    using Random
    using SequentialSamplingModels
    using StatsBase
    using Test

    Random.seed!(65)
    n_sim = 20_000
    dist = LNR(ν = [-2, -3], σ = [1.0, 1.0], τ = 0.3)
    choice, rt = rand(dist, n_sim)

    ul, ub = quantile(rt, [0.05, 0.95])
    for t ∈ range(ul, ub, length = 10)
        sim_x = 1 - mean(choice .== 1 .&& rt .≤ t)
        x = survivor(dist, 1, t)
        @test sim_x ≈ x atol = 1e-2
    end
end
