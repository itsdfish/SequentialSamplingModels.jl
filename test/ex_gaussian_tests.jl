@safetestset "ExGaussian" begin
    @safetestset "pdf 1" begin 
        using SequentialSamplingModels
        using Random
        using Test 
        include("KDE.jl")
        Random.seed!(5414)

        d = ExGaussian(μ=.5, σ=.10, τ=.20)
        rts = rand(d, 10_000)

        approx_pdf = kde(rts)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x)
        y = pdf.(d, x)
        @test y′ ≈ y rtol = .03
    end

    @safetestset "pdf 2" begin 
        using SequentialSamplingModels
        using Random
        using Test 
        include("KDE.jl")
        Random.seed!(605)

        d = ExGaussian(μ=.8, σ=.30, τ=.60)
        rts = rand(d, 20_000)

        approx_pdf = kde(rts)
        x = .2:.02:4.0
        y′ = pdf(approx_pdf, x)
        y = pdf.(d, x)
        @test y′ ≈ y rtol = .03
    end

    @safetestset "logpdf" begin 
        using SequentialSamplingModels
        using Random
        using Test 
        include("KDE.jl")
        Random.seed!(87401)

        d = ExGaussian(μ=.8, σ=.30, τ=.60)
        rts = rand(d, 10)

        LL = logpdf.(d, rts)
        like = pdf.(d, rts)

        @test exp.(LL) ≈ like 
    end

    @safetestset "mean" begin 
        using SequentialSamplingModels
        using Random
        using Test 
        Random.seed!(87401)

        d = ExGaussian(μ=.8, σ=.30, τ=.60)
        rts = rand(d, 10_000)
        sim_mean = mean(rts)
        @test mean(d) ≈ sim_mean rtol = .01
    end

    @safetestset "std" begin 
        using SequentialSamplingModels
        using Random
        using Test 
        Random.seed!(6021)

        d = ExGaussian(μ=.8, σ=.30, τ=.60)
        rts = rand(d, 20_000)
        sim_std = std(rts)
        @test std(d) ≈ sim_std rtol = .02
    end

    @safetestset "CDF" begin 
        @safetestset "1" begin 
            using Random 
            using SequentialSamplingModels
            using StatsBase
            using Test 
            
            Random.seed!(8741)
            n_sim = 10_000

            dist = ExGaussian(;μ=.5, σ=.10, τ=.20) 
            rt = rand(dist, n_sim)
            ul,ub = quantile(rt, [.05,.95])
            for t ∈ range(ul, ub, length=10) 
                sim_x = mean(rt .≤ t)
                x = cdf(dist, t)
                @test sim_x ≈ x atol = 1e-2
            end
        end

        @safetestset "2" begin 
            using Random 
            using SequentialSamplingModels
            using StatsBase
            using Test 
            
            Random.seed!(8741)
            n_sim = 10_000

            dist = ExGaussian(;μ=.6, σ=.10, τ=.40)
            rt = rand(dist, n_sim)
            ul,ub = quantile(rt, [.05,.95])
            for t ∈ range(ul, ub, length=10)
                sim_x = mean(rt .≤ t)
                x = cdf(dist, t)
                @test sim_x ≈ x atol = 1e-2
            end
        end
    end
end