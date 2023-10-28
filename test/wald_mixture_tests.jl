@safetestset "Wald Mixture" begin
    @safetestset "pdf" begin 
        using Test, SequentialSamplingModels, Random
        include("KDE.jl")
        Random.seed!(22198)
        d = WaldMixture(2, .2, 1, .1)
        @test mean(d) ≈ (1/2) + .1 atol = 1e-5
        rts = rand(d, 100000)
        approx_pdf = kde(rts)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x)
        y = pdf.(d, x)
        @test y′ ≈ y rtol = .03
        @test mean(rts) ≈ mean(d) atol = 5e-3

        y′ = @. logpdf(d, x) |> exp
        @test y′ ≈ y
    end
    
    @safetestset "loglikelihood" begin 
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(655)

        dist = WaldMixture(2, .2, 1, .1)
        rt = rand(dist, 10)

        sum_logpdf = logpdf.(dist, rt) |> sum 
        loglike = loglikelihood(dist, rt)
        @test sum_logpdf ≈ loglike 
    end

    @safetestset "simulate" begin
        using SequentialSamplingModels
        using Test
        using Random 

        Random.seed!(3233)
        α = .80
       
        dist = WaldMixture(;α)

        time_steps,evidence = simulate(dist; Δt = .0005)

        @test time_steps[1] ≈ 0
        @test length(time_steps) == length(evidence)
        @test evidence[end] ≈ α atol = .02
    end

    @safetestset "CDF" begin 
        @safetestset "1" begin 
            using Random 
            using SequentialSamplingModels
            using StatsBase
            using Test 
            
            Random.seed!(95)
            n_sim = 10_000

            dist = WaldMixture(;ν=3.0, η=.2, α=.5, τ=.130)
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
            
            Random.seed!(145)
            n_sim = 10_000

            dist = WaldMixture(;ν=2.0, η=.2, α=.8, τ=.130)
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