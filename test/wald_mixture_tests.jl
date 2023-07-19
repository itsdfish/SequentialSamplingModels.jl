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
end