@safetestset "Racing Diffusion Model" begin
    @safetestset "pdf" begin
        using SequentialSamplingModels, Test, QuadGK, Random
        import SequentialSamplingModels: WaldA
        include("KDE.jl")
        Random.seed!(741)

        dist = WaldA(ν=.5, k=.3, A=.7, θ=.2)
        rts = map(_ -> rand(dist), 1:10^6)
        approx_pdf = kernel(rts)
        x = .201:.01:2.5
        y′ = pdf(approx_pdf, x)
        y = pdf.(dist, x)
        @test mean(abs.(y .- y′)) < .02
        #@test std(abs.(y .- y′)) < .04

        p′ = quadgk(x -> pdf(dist, x), .2, Inf)[1]
        @test p′ ≈ 1 rtol = .001

        p = cdf(dist, .25)
        p′ = quadgk(x -> pdf(dist, x), .2, .25)[1]
        @test p′ ≈ p rtol = .001
        @test p ≈ mean(rts .< .25) rtol = .01

        p = cdf(dist, .5)
        p′ = quadgk(x -> pdf(dist, x), .2, .5)[1]
        @test p′ ≈ p rtol = .001
        @test p ≈ mean(rts .< .5) rtol = .01

        p = cdf(dist, .6) - cdf(dist, .3)
        p′ = quadgk(x -> pdf(dist, x), .3, .6)[1]
        @test p′ ≈ p rtol = .001
        @test p ≈ mean((rts .< .6 ).& (rts .> .3)) rtol = .01


        dist = WaldA(ν=1.0, k=.3, A=1.0, θ=.2)
        rts = map(_->rand(dist), 1:10^6)
        approx_pdf = kernel(rts)
        x = .201:.01:2.5
        y′ = pdf(approx_pdf, x)
        y = pdf.(dist, x)
        @test mean(abs.(y .- y′)) < .02
    # @test std(abs.(y .- y′)) < .04

        p′ = quadgk(x -> pdf(dist, x), .2, Inf)[1]
        @test p′ ≈ 1 rtol = .001
        
        p = cdf(dist, .25)
        p′ = quadgk(x -> pdf(dist, x), .2, .25)[1]
        @test p′ ≈ p rtol = .001
        @test p ≈ mean(rts .< .25) rtol = .01

        p = cdf(dist, .5)
        p′ = quadgk(x -> pdf(dist, x), .2, .5)[1]
        @test p′ ≈ p rtol = .001
        @test p ≈ mean(rts .< .5) rtol = .01

        p = cdf(dist, 1.0) - cdf(dist, .6)
        p′ = quadgk(x -> pdf(dist, x), .6, 1.0)[1]
        @test p′ ≈ p rtol = .001
        @test p ≈ mean((rts .< 1).& (rts .> .6)) rtol = .01

        p = cdf(dist, 1.5) - cdf(dist, 1.4)
        p′ = quadgk(x -> pdf(dist, x), 1.4, 1.5)[1]
        @test p′ ≈ p rtol = .001
        @test p ≈ mean((rts .< 1.5).& (rts .> 1.4)) rtol = .02

        dist = DiffusionRace(;ν=[1.0,.5], k=0.5, A=1.0, θ=.2)
        choice,rts = rand(dist, 10^6)
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

        p′ = quadgk(x -> pdf(dist, 1, x), .2, Inf)[1]
        @test p′ ≈ p1 rtol = .001
        
        p = quadgk(x -> pdf(dist, 1, x), .2, .3)[1]
        @test p ≈ mean(rt1 .< .3) * p1 atol = .01

        p = quadgk(x -> pdf(dist, 1, x), .2, .5)[1]
        @test p ≈ mean(rt1 .< .5) * p1 atol = .01

        p = quadgk(x -> pdf(dist, 1, x), .2, 1)[1] - quadgk(x -> pdf(dist, 1, x), .2, .6)[1]
        @test p ≈ mean((rt1 .< 1) .& (rt1 .> .6)) * p1 atol = .01

        p = quadgk(x -> pdf(dist, 1, x), .2, 1.5)[1] - quadgk(x -> pdf(dist, 1, x), .2, 1.4)[1]
        @test p ≈ mean((rt1 .< 1.5).& (rt1 .> 1.4)) atol = .01


        p′ = quadgk(x -> pdf(dist, 2, x), .2, Inf)[1]
        @test p′ ≈ p2 rtol = .001
        
        p = quadgk(x -> pdf(dist, 2, x), .2, .3)[1]
        @test p ≈ mean(rt2 .< .3) * p2 atol = .01

        p = quadgk(x -> pdf(dist, 2, x), .2, .5)[1]
        @test p ≈ mean(rt2 .< .5) * p2 atol = .02

        p = quadgk(x -> pdf(dist, 2, x), .2, 1)[1] - quadgk(x -> pdf(dist, 2, x), .2, .6)[1]
        @test p ≈ mean((rt2 .< 1) .& (rt2 .> .6)) * p2 atol = .01

        p = quadgk(x -> pdf(dist, 2, x), .2, 1.5)[1] - quadgk(x -> pdf(dist, 2, x), .2, 1.4)[1]
        @test p ≈ mean((rt2 .< 1.5).& (rt2 .> 1.4)) * p2 atol = .01
    end

    @safetestset "loglikelihood" begin 
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(655)

        dist = DiffusionRace(;ν=[1.0,.5], k=0.5, A=1.0, θ=.2)
        choice,rt = rand(dist, 10)

        sum_logpdf = logpdf.(dist, choice, rt) |> sum 
        loglike = loglikelihood(dist, (;choice, rt))
        @test sum_logpdf ≈ loglike 
    end
end