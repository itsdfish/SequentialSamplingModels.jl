@safetestset "Wald" begin
    @safetestset "pdf" begin
        using Test, SequentialSamplingModels, Random
        include("KDE.jl")
        Random.seed!(22158)
        d = Wald(2, 1, 0.1)
        @test mean(d) ≈ (1 / 2) + 0.1 atol = 1e-5


        function simulate(υ, α, τ)
            noise = 1.0
            #Time Step
            Δt = 0.0005
            #Evidence step
            Δe = noise * sqrt(Δt)
            e = 0.0
            t = τ
            p = 0.5 * (1 + υ * sqrt(Δt) / noise)
            while (e < α)
                t += Δt
                e += rand() ≤ p ? Δe : -Δe
            end
            return t
        end
        rts = map(_ -> simulate(3, 1, 0.2), 1:10^5)
        approx_pdf = kde(rts)
        x = 0.2:0.01:1.5
        y′ = pdf(approx_pdf, x)
        y = pdf.(Wald(3, 1, 0.2), x)
        @test y′ ≈ y rtol = 0.03
        @test mean(rts) ≈ mean(Wald(3, 1, 0.2)) atol = 5e-3
        @test std(rts) ≈ std(Wald(3, 1, 0.2)) atol = 1e-3
    end

    @safetestset "loglikelihood" begin
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(655)

        dist = Wald(2, 1, 0.1)
        rt = rand(dist, 10)

        sum_logpdf = logpdf.(dist, rt) |> sum
        loglike = loglikelihood(dist, rt)
        @test sum_logpdf ≈ loglike
    end

    @safetestset "simulate" begin
        using SequentialSamplingModels
        using Test
        using Random

        Random.seed!(8433)
        α = 0.80

        dist = Wald(; α)

        time_steps, evidence = simulate(dist; Δt = 0.0005)

        @test time_steps[1] ≈ 0
        @test length(time_steps) == length(evidence)
        @test evidence[end] ≈ α atol = 0.02
    end

    @safetestset "CDF" begin
        @safetestset "1" begin
            using Random
            using SequentialSamplingModels
            using StatsBase
            using Test

            Random.seed!(8741)
            n_sim = 10_000

            dist = Wald(ν = 3.0, α = 0.5, τ = 0.130)
            rt = rand(dist, n_sim)
            ul, ub = quantile(rt, [0.05, 0.95])
            for t ∈ range(ul, ub, length = 10)
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

            Random.seed!(45)
            n_sim = 10_000

            dist = Wald(ν = 2.0, α = 0.8, τ = 0.130)
            rt = rand(dist, n_sim)
            ul, ub = quantile(rt, [0.05, 0.95])
            for t ∈ range(ul, ub, length = 10)
                sim_x = mean(rt .≤ t)
                x = cdf(dist, t)
                @test sim_x ≈ x atol = 1e-2
            end
        end
    end
end
