using SafeTestsets

@safetestset "Wald" begin
    using Test, SequentialSamplingModels, KernelDensity, Random
    Random.seed!(22158)
    d = Wald(2, 1, .1)
    @test mean(d) ≈ (1/2) + .1 atol = 1e-5

    function simulate(υ, α, θ)
        noise = 1.0
        #Time Step
        Δt = .0005
        #Evidence step
        Δe = noise*sqrt(Δt)
        e = 0.0
        t = θ
        p = .5*(1 + υ*sqrt(Δt)/noise)
        while (e < α)
            t += Δt
            e += rand() ≤ p ? Δe : -Δe
        end
        return t
    end
    rts = map(_->simulate(3, 1, .2), 1:10^5)
    approx_pdf = kde(rts)
    x = .2:.01:1.5
    y′ = pdf(approx_pdf, x)
    y = pdf.(Wald(3,1,.2), x)
    @test y′ ≈ y rtol = .03
    @test mean(rts) ≈ mean(Wald(3,1,.2)) atol = 5e-3
    @test std(rts) ≈ std(Wald(3,1,.2)) atol = 1e-3
end

@safetestset "Wald Mixture" begin
    using Test, SequentialSamplingModels, KernelDensity, Random
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

@safetestset "LogNormal Race Tests" begin
    using SequentialSamplingModels, Test, Random, KernelDensity, Distributions
    Random.seed!(54054)
    d1 = LNR(;μ=[1.0], σ=1.0, ϕ=.1)
    v1 = .3
    p1 = pdf(d1, 1, v1)
    p2 = pdf(LogNormal(1, 1), v1-.1)
    @test p1 ≈ p2
    d2 = LNR(;μ=[1.0,0.0],σ=1.0,ϕ=.1)
    d3 = LNR(;μ=[1.0,0.0,0.0], σ=1.0, ϕ=.1)
    p1 = pdf(d2, 1, v1)
    p2 = pdf(d3, 1 ,v1)
    @test p1 > p2
    @test p1 ≈ pdf(LogNormal(1, 1), v1-.1)*(1-cdf(LogNormal(0, 1), v1-.1)) 

    m1,m2=-2,-1
    σ = .9
    ϕ = 0.0
    d = LNR(μ=[m1,m2], σ=σ, ϕ=ϕ)
    data = rand(d, 10^4)
    x = range(m1*.8, m1*1.2, length=100)
    y = map(x->sum(logpdf.(LNR(μ=[x,m2], σ=σ, ϕ=ϕ), data)), x)
    mv,mi = findmax(y)
    @test m1 ≈ x[mi] atol = .05

    x = range(m2*.8, m2*1.2, length=100)
    y = map(x->sum(logpdf.(LNR(μ=[m1,x], σ=σ, ϕ=ϕ), data)), x)
    mv,mi = findmax(y)
    @test m2 ≈ x[mi] atol = .05

    x = range(σ*.8, σ*1.2, length=100)
    y = map(x->sum(logpdf.(LNR(μ=[m1,m2], σ=x, ϕ=ϕ), data)), x)
    mv,mi = findmax(y)
    @test σ ≈ x[mi] atol = .05

    x = range(ϕ*.8, ϕ*1.2, length=100)
    y = map(x->sum(logpdf.(LNR(μ=[m1,m2], σ=σ, ϕ=x), data)), x)
    mv,mi = findmax(y)
    @test ϕ ≈ x[mi] atol = .0005

    d = LNR(μ=[m1,m2], σ=σ, ϕ=ϕ)
    data = rand(d, 10^4)
    y1 = map(x->logpdf(d, x), data)
    y2 = map(x->log(pdf(d, x)), data)
    @test y1 ≈ y2 atol = .00001

    d = LNR(μ=[1.,.5], σ=.4, ϕ=.5)
    data = rand(d, 10^4)
    y1 = map(x->logpdf(d, x), data)
    y2 = map(x->log(pdf(d, x)), data)
    @test y1 ≈ y2 atol = .00001

    dist = LNR(μ=[-1.5,-.9], σ=.5, ϕ=.3) 
    data = rand(dist, 10^5)
    data1 = filter(x->x[1] == 1, data)
    p1 = mean(x->x[1] == 1, data)
    p2 = 1 - p1
    rt1 = map(x->x[2], data1)
    approx_pdf = kde(rt1)
    x = .2:.01:1.5
    y′ = pdf(approx_pdf, x)*p1
    y = pdf.(dist, (1,), x)
    @test y′ ≈ y rtol = .03

    data2 = filter(x->x[1] == 2, data)
    rt2 = map(x->x[2], data2)
    approx_pdf = kde(rt2)
    x = .2:.01:1.5
    y′ = pdf(approx_pdf, x)*p2
    y = pdf.(dist, (2,), x)
    @test y′ ≈ y rtol = .03
end

@safetestset "LBA Tests" begin
    using SequentialSamplingModels, Test, KernelDensity, Random
    Random.seed!(10542)
    dist = LBA(ν=[3.0,2.0], A = .8, k = .2, τ = .3) 
    choice,rt = rand(dist, 10^5)
    rt1 = rt[choice .== 1]
    p1 = mean(x->x==1, choice)
    p2 = 1 - p1
    approx_pdf = kde(rt1)
    x = .2:.01:1.5
    y′ = pdf(approx_pdf, x)*p1
    y = pdf.(dist, (1,), x)
    @test y′ ≈ y rtol = .03

    rt2 = rt[choice .== 2]
    approx_pdf = kde(rt2)
    x = .2:.01:1.5
    y′ = pdf(approx_pdf, x)*p2
    y = pdf.(dist, (2,), x)
    @test y′ ≈ y rtol = .03
end