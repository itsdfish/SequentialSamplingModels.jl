@safetestset "DDM Tests" begin
    @safetestset "DDM pdf 1" begin
        using SequentialSamplingModels
        using Test
        using KernelDensity
        using Random
        Random.seed!(654)

        dist = DDM(ν=1.0, α = .8, z = .5, τ = .3) 
        choice,rt = rand(dist, 10^5)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        approx_pdf = kde(rt1)
        x = range(.301, 1.5, length=100)
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .10

        rt2 = rt[choice .== 2]
        approx_pdf = kde(rt2)
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .10
    end

    @safetestset "DDM pdf 2" begin
        using SequentialSamplingModels
        using Test
        using KernelDensity
        using Random
        Random.seed!(750)

        dist = DDM(ν=2.0, α = 1.5, z = .5, τ = .30) 
        choice,rt = rand(dist, 10^5)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        approx_pdf = kde(rt1)
        x = range(.301, 1.5, length=100)
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .05

        rt2 = rt[choice .== 2]
        approx_pdf = kde(rt2)
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .10
    end

    @safetestset "DDM cdf 1" begin
        using SequentialSamplingModels
        using Test
        using StatsBase
        using Random
        Random.seed!(7540)

        dist = DDM(ν=1.0, α = .8, z = .5, τ = .3) 
        choice,rt = rand(dist, 10^5)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        ecdf1 = ecdf(rt1)
        x = range(.31, 1.0, length=100)
        y′ = ecdf1.(x) * p1
        y = cdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .01

        rt2 = rt[choice .== 2]
        ecdf2 = ecdf(rt2)
        y′ = ecdf1.(x) * p2
        y = cdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .01
    end

    @safetestset "DDM cdf 2" begin
        using SequentialSamplingModels
        using Test
        using StatsBase
        using Random
        Random.seed!(2200)

        dist = DDM(ν=2.0, α = 1.5, z = .5, τ = .30) 
        choice,rt = rand(dist, 10^5)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        ecdf1 = ecdf(rt1)
        x = range(.31, 1.0, length=100)
        y′ = ecdf1.(x) * p1
        y = cdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .01

        rt2 = rt[choice .== 2]
        ecdf2 = ecdf(rt2)
        y′ = ecdf1.(x) * p2
        y = cdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .01
    end
end