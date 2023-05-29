@safetestset "LBA Tests" begin
    @safetestset "LBA Test1" begin
        using SequentialSamplingModels, Test, KernelDensity, Random
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
        using SequentialSamplingModels, Test, KernelDensity, Random
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

    @safetestset "LBA loglikelihood" begin
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(8521)

        dist = LBA(ν=[2.0,2.7], A = .6, k = .26, τ = .4) 
        choice,rt = rand(dist, 10)

        sum_logpdf = logpdf.(dist, choice, rt) |> sum 
        loglike = loglikelihood(dist, (choice, rt))
        @test sum_logpdf ≈ loglike 
    end
end