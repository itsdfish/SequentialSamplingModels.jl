@safetestset "LBA Tests" begin
    @safetestset "LBA Test1" begin
        using SequentialSamplingModels, Test, Random
        include("KDE.jl")
        Random.seed!(10542)

        dist = LBA(ν=[3.0,2.0], A = .8, k = .2, τ = .3) 
        choice,rt = rand(dist, 10^5)
        rt1 = rt[choice .== 1]
        p1 = mean(x -> x == 1, choice)
        p2 = 1 - p1
        approx_pdf = kde(rt1)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .03

        rt2 = rt[choice .== 2]
        approx_pdf = kde(rt2)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .03
    end

    @safetestset "LBA Test2" begin
        using SequentialSamplingModels, Test, Random
        include("KDE.jl")
        Random.seed!(8521)

        # note for some values, tests will fail
        # this is because kde is sensitive to outliers
        # density overlay on histograms are valid
        dist = LBA(ν=[2.0,2.7], A = .6, k = .26, τ = .4) 
        choice,rt = rand(dist, 10^5)
        rt1 = rt[choice .== 1]
        p1 = mean(x -> x == 1, choice)
        p2 = 1 - p1
        approx_pdf = kde(rt1)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .03

        rt2 = rt[choice .== 2]
        approx_pdf = kde(rt2)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .03
    end

    @safetestset "LBA Test3" begin
        using SequentialSamplingModels, Test, Random
        include("KDE.jl")
        Random.seed!(851)

        # note for some values, tests will fail
        # this is because kde is sensitive to outliers
        # density overlay on histograms are valid
        dist = LBA(ν=[2.0,2.7], A = .4, k = .20, τ = .4, σ=[1.0,0.5]) 
        choice,rt = rand(dist, 10^5)
        rt1 = rt[choice .== 1]
        p1 = mean(x -> x == 1, choice)
        p2 = 1 - p1
        approx_pdf = kde(rt1)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .03

        rt2 = rt[choice .== 2]
        approx_pdf = kde(rt2)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .03
    end

    @safetestset "LBA loglikelihood" begin
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(8521)

        dist = LBA(ν=[2.0,2.7], A = .6, k = .26, τ = .4) 
        choice,rt = rand(dist, 10)

        sum_logpdf = logpdf.(dist, choice, rt) |> sum 
        loglike = loglikelihood(dist, (;choice, rt))
        @test sum_logpdf ≈ loglike 
    end

    @safetestset "simulate" begin
        using SequentialSamplingModels
        using Test
        using Random 

        Random.seed!(8477)
        A = .80
        k = .20
        α = A + k
        dist = LBA(;A, k, ν=[2,1])

        time_steps,evidence = simulate(dist; n_steps = 100)

        @test time_steps[1] ≈ 0
        @test length(time_steps) == size(evidence, 1)
        @test size(evidence, 2) == 2
        @test maximum(evidence[end,:]) ≈ α atol = .005
    end
end