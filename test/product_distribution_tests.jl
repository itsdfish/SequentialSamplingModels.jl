@safetestset "ProductDistribution Tests" begin
    @safetestset "rand SSM1D 1" begin
        using Distributions
        using SequentialSamplingModels
        using Test

        walds = [Wald(; ν = 2.5, α = 0.1, τ = 0.2), Wald(; ν = 1.5, α = 1, τ = 10)]
        dist = product_distribution(walds)
        data = rand(dist)
        @test length(data) == 2
        @test data[1] < 10
        @test data[2] > 10
    end

    @safetestset "rand SSM1D 2" begin
        using Distributions
        using SequentialSamplingModels
        using Test

        walds = [Wald(; ν = 2.5, α = 0.1, τ = 0.2), Wald(; ν = 1.5, α = 1, τ = 10)]
        dist = product_distribution(walds)
        data = rand(dist, 3)
        @test size(data) == (2, 3)
        @test all(data[1, :] .< 10)
        @test all(data[2, :] .> 10)
    end

    @safetestset "rand logpdf 1" begin
        using Distributions
        using SequentialSamplingModels
        using Test

        walds = [Wald(; ν = 2.5, α = 0.1, τ = 0.2), Wald(; ν = 1.5, α = 1, τ = 10)]
        dist = product_distribution(walds)
        data = rand(dist)
        LL1 = logpdf(dist, data)
        LL2 = sum(i -> logpdf(walds[i], data[i]), 1:2)
        @test LL1 ≈ LL2
    end

    @safetestset "logpdf SSM1D 2" begin
        using Distributions
        using SequentialSamplingModels
        using Test

        walds = [Wald(; ν = 2.5, α = 0.1, τ = 0.2), Wald(; ν = 1.5, α = 1, τ = 10)]
        dist = product_distribution(walds)
        data = rand(dist, 3)
        LL1 = logpdf(dist, data)
        LL2 = sum(i -> logpdf(walds[i], data[i, :]), 1:2)
        @test LL1 ≈ LL2
    end

    @safetestset "rand SSM2D 1" begin
        using Distributions
        using SequentialSamplingModels
        using Test

        lbas = [
            LBA(; ν = [3, 2], A = 0.8, k = 0.2, τ = 0.1),
            LBA(ν = [1, 2], A = 0.5, k = 0.3, τ = 10)
        ]
        dist = product_distribution(lbas)
        data = rand(dist)
        @test length(data.rt) == 2
        @test data.rt[1] < 10
        @test data.rt[2] > 10
    end

    @safetestset "rand SSM2D 2" begin
        using Distributions
        using SequentialSamplingModels
        using Test

        lbas = [
            LBA(; ν = [3, 2], A = 0.8, k = 0.2, τ = 0.1),
            LBA(ν = [1, 2], A = 0.5, k = 0.3, τ = 10)
        ]
        dist = product_distribution(lbas)
        data = rand(dist, 3)
        @test length(data) == 3
        @test all(map(i -> data[i].rt[1], 1:3) .< 10)
        @test all(map(i -> data[i].rt[2], 1:3) .> 10)
    end

    @safetestset "logpdf SSM2D 1" begin
        using Distributions
        using SequentialSamplingModels
        using Test

        lbas = [
            LBA(; ν = [3, 2], A = 0.8, k = 0.2, τ = 0.1),
            LBA(ν = [1, 2], A = 0.5, k = 0.3, τ = 10)
        ]
        dist = product_distribution(lbas)
        data = rand(dist)

        LL1 = logpdf(dist, data)
        LL2 = sum(i -> logpdf(lbas[i], data.choice[i], data.rt[i]), 1:2)
        @test LL1 ≈ LL2
    end

    @safetestset "logpdf SSM2D 2" begin
        using Distributions
        using SequentialSamplingModels
        using Test

        lbas = [
            LBA(; ν = [3, 2], A = 0.8, k = 0.2, τ = 0.1),
            LBA(ν = [1, 2], A = 0.5, k = 0.3, τ = 10)
        ]
        dist = product_distribution(lbas)
        data = rand(dist, 3)

        LL1 = logpdf(dist, data)
        LL2 =
            map(j -> sum(i -> logpdf(lbas[i], data[j].choice[i], data[j].rt[i]), 1:2), 1:3)
        @test LL1 ≈ LL2
    end
end
