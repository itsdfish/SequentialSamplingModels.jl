@safetestset "Poisson Race Tests" begin
    @safetestset "pdf1" begin 
        using SequentialSamplingModels, Test, Random, Distributions
        include("KDE.jl")
        Random.seed!(5364)

        dist = PoissonRace(;ν=[.05,.06], α=[4,5], τ=.3)
        choice,rts = rand(dist, 10^5)
        rts1 = rts[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        approx_pdf = kde(rts1)
        x = .3:.01:1.5
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .02

        rts2 = rts[choice .== 2]
        approx_pdf = kde(rts2)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .02
    end

    @safetestset "pdf2" begin 
        using SequentialSamplingModels, Test, Random, Distributions
        include("KDE.jl")
        Random.seed!(65)

        dist = PoissonRace(;ν=[.04,.045], α=[4,3], τ=.2)
        choice,rts = rand(dist, 10^5)
        rts1 = rts[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        approx_pdf = kde(rts1)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .02

        rts2 = rts[choice .== 2]
        approx_pdf = kde(rts2)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .02
    end

    @safetestset "loglikelihood" begin
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(974)

        dist = PoissonRace(;ν=[.05,.06], α=[4,5], τ=.3)
        choice,rt = rand(dist, 10)

        sum_logpdf = logpdf.(dist, choice, rt) |> sum 
        loglike = loglikelihood(dist, (;choice, rt))
        @test sum_logpdf ≈ loglike 
    end
end