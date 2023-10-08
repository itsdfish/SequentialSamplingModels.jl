@safetestset "compute_quantiles" begin 
    @safetestset "SSM2D" begin 
        using SequentialSamplingModels
        using Test 

        dist = LBA(ν=[3.0,3.0], A = .8, k = .2, τ = .3) 
        data = rand(dist, 100)
        percentiles = [.3,.5,.7]
        qs = compute_quantiles(data; percentiles)
        @test length.(qs) == [3,3]
        
        qs = compute_quantiles(data; percentiles, choice_set=[:a,:b])
        @test length(qs) == 2
        @test isempty(qs[1])
        @test isempty(qs[2])
    end

    @safetestset "ContinuousMultivariateSSM" begin 
        using SequentialSamplingModels
        using Test 

        dist = CDDM(;ν=[1,.5], η=[1,1], σ=1, α=1.5, τ=0.30)
        data = rand(dist, 10)
        percentiles = [.3,.5,.7]
        qs = compute_quantiles(data; percentiles)
        @test length.(qs) == [3,3]
    end

    @safetestset "1D SSM" begin 
        using SequentialSamplingModels
        using Test 

        dist = Wald(ν=3.0, α=.5, τ=.130)
        rts = rand(dist, 10)
        percentiles = [.3,.5,.7]
        qs = compute_quantiles(rts; percentiles)
        @test length(qs) == 3
    end
end

@safetestset "compute_choice_probs" begin 
    using SequentialSamplingModels
    using Test 

    data = (choice=[1,1,2],rts=[.3,.5,.3])
    choice_probs = compute_choice_probs(data; choice_set=[1,2])

    @test choice_probs == [2/3, 1/3]
end
