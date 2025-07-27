@safetestset "ExGaussian" begin
    @safetestset "pdf 1" begin
        using SequentialSamplingModels
        using StableRNGs
        using Test
        include("../KDE.jl")
        rng = StableRNG(584)

        d = ExGaussian(μ = 0.5, σ = 0.10, τ = 0.20)
        rts = rand(rng, d, 10_000)

        approx_pdf = kde(rts)
        x = 0.2:0.01:1.5
        y′ = pdf(approx_pdf, x)
        y = pdf.(d, x)
        @test y′ ≈ y rtol = 0.03
    end

    @safetestset "pdf 2" begin
        using SequentialSamplingModels
        using StableRNGs
        using Test
        include("../KDE.jl")
        rng = StableRNG(123)

        d = ExGaussian(μ = 0.8, σ = 0.30, τ = 0.60)
        rts = rand(rng, d, 20_000)

        approx_pdf = kde(rts)
        x = 0.2:0.02:4.0
        y′ = pdf(approx_pdf, x)
        y = pdf.(d, x)
        @test y′ ≈ y rtol = 0.03
    end

    @safetestset "pdf 3" begin
        using Distributions
        using StableRNGs
        using SequentialSamplingModels
        using SequentialSamplingModels: Φ
        using Test
        rng = StableRNG(345)

        # direct pdf, not used due to numerical instability in edge cases
        function _pdf(d::ExGaussian, rt::Float64)
            (; μ, σ, τ) = d
            return (1 / τ) * exp((μ - rt) / τ + (σ^2 / 2τ^2)) * Φ((rt - μ) / σ - (σ / τ))
        end

        for _ ∈ 1:100
            d = ExGaussian(rand(rng, Uniform(0.3, 1), 3)...)
            rts = rand(rng, d, 10)
            @test pdf.(d, rts) ≈ _pdf.(d, rts)
        end
    end

    @safetestset "logpdf" begin
        using SequentialSamplingModels
        using StableRNGs
        using Test
        include("../KDE.jl")
        rng = StableRNG(334)

        d = ExGaussian(μ = 0.8, σ = 0.30, τ = 0.60)
        rts = rand(rng, d, 10)

        LL = logpdf.(d, rts)
        like = pdf.(d, rts)

        @test exp.(LL) ≈ like
    end

    @safetestset "mean" begin
        using SequentialSamplingModels
        using StableRNGs
        using Test
        rng = StableRNG(45)

        d = ExGaussian(μ = 0.8, σ = 0.30, τ = 0.60)
        rts = rand(rng, d, 10_000)
        sim_mean = mean(rts)
        @test mean(d) ≈ sim_mean rtol = 0.01
    end

    @safetestset "std" begin
        using SequentialSamplingModels
        using StableRNGs
        using Test
        rng = StableRNG(3456)

        d = ExGaussian(μ = 0.8, σ = 0.30, τ = 0.60)
        rts = rand(rng, d, 20_000)
        sim_std = std(rts)
        @test std(d) ≈ sim_std rtol = 0.02
    end

    @safetestset "CDF" begin
        @safetestset "1" begin
            using StableRNGs
            using SequentialSamplingModels
            using StatsBase
            using Test

            rng = StableRNG(333)
            n_sim = 10_000

            dist = ExGaussian(; μ = 0.5, σ = 0.10, τ = 0.20)
            rt = rand(rng, dist, n_sim)
            ul, ub = quantile(rt, [0.05, 0.95])
            for t ∈ range(ul, ub, length = 10)
                sim_x = mean(rt .≤ t)
                x = cdf(dist, t)
                @test sim_x ≈ x atol = 1e-2
            end
        end

        @safetestset "2" begin
            using StableRNGs
            using SequentialSamplingModels
            using StatsBase
            using Test

            rng = StableRNG(122)
            n_sim = 10_000

            dist = ExGaussian(; μ = 0.6, σ = 0.10, τ = 0.40)
            rt = rand(rng, dist, n_sim)
            ul, ub = quantile(rt, [0.05, 0.95])
            for t ∈ range(ul, ub, length = 10)
                sim_x = mean(rt .≤ t)
                x = cdf(dist, t)
                @test sim_x ≈ x atol = 1e-2
            end
        end
    end

    @safetestset "params" begin
        using Test
        using Distributions
        using SequentialSamplingModels

        parms = (; μ = 0.5, σ = 0.20, τ = 0.20)

        model = ExGaussian(; parms...)
        @test values(parms) == params(model)
    end

    @safetestset "parameter checks" begin
        @safetestset "all valid" begin
            using Test
            using Distributions
            using SequentialSamplingModels

            parms = (; μ = 0.5, σ = 0.20, τ = 0.20)

            ExGaussian(; parms...)
            ExGaussian(values(parms)...)
            @test true
        end

        @safetestset "μ invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels

            parms = (; μ = -1, σ = 0.20, τ = 0.20)

            @test_throws ArgumentError ExGaussian(; parms...)
            @test_throws ArgumentError ExGaussian(values(parms)...)
        end

        @safetestset "σ invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels

            parms = (; μ = 0.5, σ = -0.20, τ = 0.20)

            @test_throws ArgumentError ExGaussian(; parms...)
            @test_throws ArgumentError ExGaussian(values(parms)...)
        end

        @safetestset "τ invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels

            parms = (; μ = 0.5, σ = 0.20, τ = -0.20)

            @test_throws ArgumentError ExGaussian(; parms...)
            @test_throws ArgumentError ExGaussian(values(parms)...)
        end
    end
end
