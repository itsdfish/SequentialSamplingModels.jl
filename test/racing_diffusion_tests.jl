@safetestset "Racing Diffusion Model" begin
    @safetestset "pdf" begin
        using SequentialSamplingModels, Test, QuadGK, Random
        import SequentialSamplingModels: WaldA
        include("KDE.jl")
        Random.seed!(741)

        dist = WaldA(ν = 0.5, k = 0.3, A = 0.7, τ = 0.2)
        rts = map(_ -> rand(dist), 1:(10^6))
        approx_pdf = kernel(rts)
        x = 0.201:0.01:2.5
        y′ = pdf(approx_pdf, x)
        y = pdf.(dist, x)
        @test mean(abs.(y .- y′)) < 0.02

        p′ = quadgk(x -> pdf(dist, x), 0.2, Inf)[1]
        @test p′ ≈ 1 rtol = 0.001

        p = cdf(dist, 0.25)
        p′ = quadgk(x -> pdf(dist, x), 0.2, 0.25)[1]
        @test p′ ≈ p rtol = 0.001
        @test p ≈ mean(rts .< 0.25) rtol = 0.01

        p = cdf(dist, 0.5)
        p′ = quadgk(x -> pdf(dist, x), 0.2, 0.5)[1]
        @test p′ ≈ p rtol = 0.001
        @test p ≈ mean(rts .< 0.5) rtol = 0.01

        p = cdf(dist, 0.6) - cdf(dist, 0.3)
        p′ = quadgk(x -> pdf(dist, x), 0.3, 0.6)[1]
        @test p′ ≈ p rtol = 0.001
        @test p ≈ mean((rts .< 0.6) .& (rts .> 0.3)) rtol = 0.01

        dist = WaldA(ν = 1.0, k = 0.3, A = 1.0, τ = 0.2)
        rts = map(_ -> rand(dist), 1:(10^6))
        approx_pdf = kernel(rts)
        x = 0.201:0.01:2.5
        y′ = pdf(approx_pdf, x)
        y = pdf.(dist, x)
        @test mean(abs.(y .- y′)) < 0.02

        p′ = quadgk(x -> pdf(dist, x), 0.2, Inf)[1]
        @test p′ ≈ 1 rtol = 0.001

        p = cdf(dist, 0.25)
        p′ = quadgk(x -> pdf(dist, x), 0.2, 0.25)[1]
        @test p′ ≈ p rtol = 0.001
        @test p ≈ mean(rts .< 0.25) rtol = 0.01

        p = cdf(dist, 0.5)
        p′ = quadgk(x -> pdf(dist, x), 0.2, 0.5)[1]
        @test p′ ≈ p rtol = 0.001
        @test p ≈ mean(rts .< 0.5) rtol = 0.01

        p = cdf(dist, 1.0) - cdf(dist, 0.6)
        p′ = quadgk(x -> pdf(dist, x), 0.6, 1.0)[1]
        @test p′ ≈ p rtol = 0.001
        @test p ≈ mean((rts .< 1) .& (rts .> 0.6)) rtol = 0.01

        p = cdf(dist, 1.5) - cdf(dist, 1.4)
        p′ = quadgk(x -> pdf(dist, x), 1.4, 1.5)[1]
        @test p′ ≈ p rtol = 0.001
        @test p ≈ mean((rts .< 1.5) .& (rts .> 1.4)) rtol = 0.02

        dist = RDM(; ν = [1.0, 0.5], k = 0.5, A = 1.0, τ = 0.2)
        choice, rts = rand(dist, 10^6)
        rt1 = rts[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        # approx_pdf = kernel(rt1)
        # x = .201:.01:2.5
        # y′ = pdf(approx_pdf, x) * p1
        # y = pdf.(dist, (1,), x)
        # @test y′ ≈ y rtol = .03

        rt2 = rts[choice .== 2]
        approx_pdf = kernel(rt2)
        # x = .201:.01:1.5
        # y′ = pdf(approx_pdf, x) * p2
        # y = pdf.(dist, (2,), x)
        # @test y′ ≈ y rtol = .03

        p′ = quadgk(x -> pdf(dist, 1, x), 0.2, Inf)[1]
        @test p′ ≈ p1 rtol = 0.001

        p = quadgk(x -> pdf(dist, 1, x), 0.2, 0.3)[1]
        @test p ≈ mean(rt1 .< 0.3) * p1 atol = 0.01

        p = quadgk(x -> pdf(dist, 1, x), 0.2, 0.5)[1]
        @test p ≈ mean(rt1 .< 0.5) * p1 atol = 0.01

        p =
            quadgk(x -> pdf(dist, 1, x), 0.2, 1)[1] -
            quadgk(x -> pdf(dist, 1, x), 0.2, 0.6)[1]
        @test p ≈ mean((rt1 .< 1) .& (rt1 .> 0.6)) * p1 atol = 0.01

        p =
            quadgk(x -> pdf(dist, 1, x), 0.2, 1.5)[1] -
            quadgk(x -> pdf(dist, 1, x), 0.2, 1.4)[1]
        @test p ≈ mean((rt1 .< 1.5) .& (rt1 .> 1.4)) atol = 0.01

        p′ = quadgk(x -> pdf(dist, 2, x), 0.2, Inf)[1]
        @test p′ ≈ p2 rtol = 0.001

        p = quadgk(x -> pdf(dist, 2, x), 0.2, 0.3)[1]
        @test p ≈ mean(rt2 .< 0.3) * p2 atol = 0.01

        p = quadgk(x -> pdf(dist, 2, x), 0.2, 0.5)[1]
        @test p ≈ mean(rt2 .< 0.5) * p2 atol = 0.02

        p =
            quadgk(x -> pdf(dist, 2, x), 0.2, 1)[1] -
            quadgk(x -> pdf(dist, 2, x), 0.2, 0.6)[1]
        @test p ≈ mean((rt2 .< 1) .& (rt2 .> 0.6)) * p2 atol = 0.01

        p =
            quadgk(x -> pdf(dist, 2, x), 0.2, 1.5)[1] -
            quadgk(x -> pdf(dist, 2, x), 0.2, 1.4)[1]
        @test p ≈ mean((rt2 .< 1.5) .& (rt2 .> 1.4)) * p2 atol = 0.01
    end

    @safetestset "loglikelihood" begin
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(655)

        dist = RDM(; ν = [1.0, 0.5], k = 0.5, A = 1.0, τ = 0.2)
        choice, rt = rand(dist, 10)

        sum_logpdf = logpdf.(dist, choice, rt) |> sum
        loglike = loglikelihood(dist, (; choice, rt))
        @test sum_logpdf ≈ loglike
    end

    @safetestset "simulate" begin
        using SequentialSamplingModels
        using Test
        using Random

        Random.seed!(8477)
        A = 0.80
        k = 0.20
        α = A + k
        dist = RDM(; A, k, ν = [2, 1])

        time_steps, evidence = simulate(dist; Δt = 0.0001)

        @test time_steps[1] ≈ 0
        @test length(time_steps) == size(evidence, 1)
        @test size(evidence, 2) == 2
        @test maximum(evidence[end, :]) ≈ α atol = 0.005
    end

    @safetestset "CDF" begin
        @safetestset "1" begin
            using Random
            using SequentialSamplingModels
            using StatsBase
            using Test

            Random.seed!(602)
            n_sim = 20_000
            dist = RDM(; ν = [1, 2], k = 0.3, A = 0.7, τ = 0.2)
            choice, rt = rand(dist, n_sim)
            ul, ub = quantile(rt, [0.05, 0.95])
            for t ∈ range(ul, ub, length = 10)
                sim_x = mean(choice .== 1 .&& rt .≤ t)
                x = cdf(dist, 1, t)
                @test sim_x ≈ x atol = 1e-2
            end
        end

        @safetestset "2" begin
            using Random
            using SequentialSamplingModels
            using StatsBase
            using Test

            Random.seed!(111)
            n_sim = 20_000
            dist = RDM(; ν = [0.3, 0.4], k = 0.1, A = 0.3, τ = 0.2)
            choice, rt = rand(dist, n_sim)
            ul, ub = quantile(rt, [0.05, 0.95])
            for t ∈ range(ul, ub, length = 10)
                sim_x = mean(choice .== 1 .&& rt .≤ t)
                x = cdf(dist, 1, t)
                @test sim_x ≈ x atol = 1e-2
            end
        end
    end
end
