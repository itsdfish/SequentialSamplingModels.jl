@safetestset "Wald Mixture" begin
    @safetestset "pd" begin 
        @safetestset "pdf 1" begin 
            using Test, SequentialSamplingModels, Random
            include("KDE.jl")
            Random.seed!(54)
            d = WaldMixture(3, .5, 1, .1)
            @test mean(d) ≈ (1/3) + .1 atol = 1e-5
            rts = rand(d, 200_000)
            # remove outliers
            filter!(x -> x < 5, rts)
            approx_pdf = kde(rts)
            x = .11:.01:1.5
            y′ = pdf(approx_pdf, x)
            y = pdf.(d, x)
            @test y′ ≈ y rtol = .03
            @test mean(rts) ≈ mean(d) atol = 5e-2
        end

        @safetestset "pdf 2" begin 
            using Test
            using SequentialSamplingModels
            using QuadGK
        
            dist = WaldMixture(;ν=3.0, η=1.0, α=.5, τ=.130)
            integral,_ = quadgk(t -> pdf(dist, t), .130, 20.0)
            @test integral ≈ 1 atol = 1e-4

            dist = WaldMixture(;ν=2.0, η=.5, α=.3, τ=.130)
            integral,_ = quadgk(t -> pdf(dist, t), .130, 20.0)
            @test integral ≈ 1 atol = 1e-4
        end
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

    @safetestset "logpdf" begin
        using SequentialSamplingModels
        using Test
        using Random 

        dist = WaldMixture(;ν=2.0, η=.5, α=.3, τ=.130)
        t = range(.131, 2, length=20)
        
        pdfs = pdf.(dist, t)
        logpdfs = logpdf.(dist, t)

        @test logpdfs ≈ log.(pdfs) atol = 1e-8
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

            dist = WaldMixture(;ν=3.0, η=.5, α=.5, τ=.130)
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