@safetestset "LogNormal Race Tests" begin
    @safetestset "pdf" begin 
        using SequentialSamplingModels, Test, Random, Distributions
        include("KDE.jl")
        Random.seed!(54154)
        d1 = LNR(;ν=[1.0], σ=[1.0], τ=.1)
        v1 = .3
        p1 = pdf(d1, 1, v1)
        p2 = pdf(LogNormal(1, 1), v1-.1)
        @test p1 ≈ p2
        d2 = LNR(;ν=[1.0,0.0],σ=[1.0,1.0],τ=.1)
        d3 = LNR(;ν=[1.0,0.0,0.0], σ=[1.0,1.0,1.0], τ=.1)
        p1 = pdf(d2, 1, v1)
        p2 = pdf(d3, 1 ,v1)
        @test p1 > p2
        @test p1 ≈ pdf(LogNormal(1, 1), v1-.1)*(1-cdf(LogNormal(0, 1), v1-.1)) 

        m1,m2=-2,-1
        σ = fill(.9, 2)
        τ = 0.0
        d = LNR(ν=[m1,m2], σ=σ, τ=τ)
        choice,rts = rand(d, 10^4)
        x = range(m1*.8, m1*1.2, length=100)
        y = map(x -> sum(logpdf.(LNR(ν=[x,m2], σ=σ, τ=τ), choice, rts)), x)
        mv,mi = findmax(y)
        @test m1 ≈ x[mi] atol = .05

        x = range(m2*.8, m2*1.2, length=100)
        y = map(x -> sum(logpdf.(LNR(ν=[m1,x], σ=σ, τ=τ), choice, rts)), x)
        mv,mi = findmax(y)
        @test m2 ≈ x[mi] atol = .05

        x = range(σ*.8, σ*1.2, length=100)
        y = map(x -> sum(logpdf.(LNR(ν=[m1,m2], σ=x, τ=τ), choice, rts)), x)
        mv,mi = findmax(y)
        @test σ ≈ x[mi] atol = .05

        x = range(τ*.8, τ*1.2, length=100)
        y = map(x -> sum(logpdf.(LNR(ν=[m1,m2], σ=σ, τ=x), choice, rts)), x)
        mv,mi = findmax(y)
        @test τ ≈ x[mi] atol = .0005

        d = LNR(ν=[m1,m2], σ=σ, τ=τ)
        choice,rts  = rand(d, 10^4)
        y1 = logpdf.(d, choice, rts)
        y2 = log.(pdf.(d, choice, rts))
        @test y1 ≈ y2 atol = .00001

        d = LNR(ν=[1.,.5], σ=[.4,.4], τ=.5)
        choice,rts  = rand(d, 10^4)
        y1 = logpdf.(d, choice, rts)
        y2 = log.(pdf.(d, choice, rts))
        @test y1 ≈ y2 atol = .00001

        dist = LNR(ν=[-1.5,-.9], σ=[.5,.5], τ=.3) 
        choice,rts = rand(dist, 10^5)
        rts1 = rts[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        approx_pdf = kde(rts1)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .03

        rts2 = rts[choice .== 2]
        approx_pdf = kde(rts2)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .03
    end

    @safetestset "LNR loglikelihood" begin
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(8521)

        dist = LNR(;ν=[1.0,.5], σ=[1.0,1.0], τ=.1)
        choice,rt = rand(dist, 10)

        sum_logpdf = logpdf.(dist, choice, rt) |> sum 
        loglike = loglikelihood(dist, (;choice, rt))
        @test sum_logpdf ≈ loglike 
    end
end