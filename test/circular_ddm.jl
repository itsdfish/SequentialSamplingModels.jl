@safetestset "CDDM" begin
    @safetestset "bessel_hm" begin 
        using Test 
        using SequentialSamplingModels
        using SequentialSamplingModels: bessel_hm 

        model = CDDM(;
            ν = [1.5,.5],
            η = [1,1],
            σ = 1.0,
            α = 1.5,
            τ = 0.0)

        rts = [0:.1:1;]
        ground_truth = [0., 0.00286158, 0.19383708, 0.54859974, 0.76896096,
            0.84213747, 0.82883946, 0.7744823 , 0.7043917 , 0.63131337, 0.56121361]

        densities = map(rt -> bessel_hm(model, rt), rts) * (2 * π)
        @test densities ≈ ground_truth 
    end

    @safetestset "bessel_s" begin 
        using Test 
        using SequentialSamplingModels
        using SequentialSamplingModels: bessel_s

        model = CDDM(;
            ν = [1.5,.5],
            η = [1,1],
            σ = 1.0,
            α = 1.5,
            τ = 0.0)

        rts = [0:.1:1;]
        ground_truth = [0. , 0.00280757, 0.18601942, 0.51358442, 0.70062069,
            0.74526978, 0.71124981, 0.64357356, 0.56624328, 0.49064167,0.4215687] 

        densities = map(rt -> bessel_s(model, rt), rts)
        @test densities ≈ ground_truth atol = 1e-5
    end

    @safetestset "rand" begin 
        @safetestset "rand 1" begin 
            using Test 
            using Distributions
            using Random
            using SequentialSamplingModels
            using Statistics 
            include("KDE.jl")

            Random.seed!(5)

            model = CDDM(;
                ν = [1.5,1],
                η = [0.0,0.0],
                σ = 1.0,
                α = 1.5,
                τ = .300)

            data = rand(model, 100_000)
            approx_pdf = kernel(data[:,1])

            x = range(-π, π, length = 100)

            μ = atan(model.ν[2], model.ν[1])
            κ = model.α * √(sum(model.ν.^2)) / model.σ^2
            y_true = pdf.(VonMises(μ, κ), x)
            y = pdf(approx_pdf, x)
            @test y ≈ y_true rtol = .015
            @test maximum(abs.(y_true .- y)) < .02
        end

        @safetestset "rand 2" begin 
            using Test 
            using Distributions
            using Random
            using SequentialSamplingModels
            using Statistics 
            include("KDE.jl")

            Random.seed!(145)

            model = CDDM(;
                ν = [1.5,-1],
                η = [0.0,0.0],
                σ = .5,
                α = 2.5,
                τ = .400)

            data = rand(model, 100_000)
            approx_pdf = kernel(data[:,1])

            x = range(-π, π, length = 200)

            μ = atan(model.ν[2], model.ν[1])
            κ = model.α * √(sum(model.ν.^2)) / model.σ^2
            y_true = pdf.(VonMises(μ, κ), x)
            y = pdf(approx_pdf, x)
            @test y ≈ y_true rtol = .015
            @test maximum(abs.(y_true .- y)) < .02
        end
    end

    @safetestset "pdf_rt" begin 
        @safetestset "pdf_rt 1" begin 
            using Test 
            using Distributions
            using Random
            using SequentialSamplingModels
            using SequentialSamplingModels: pdf_rt 
            using Statistics 
            include("KDE.jl")
    
            Random.seed!(1345)
    
            model = CDDM(;
                ν=[1.75,1.0],
                η = [.50,.50],
                σ = .50,
                α = 2.5,
                τ = .20)

            rts = range(model.τ, 3.5, length=200)
            dens = map(rt -> pdf_rt(model, rt), rts)
            data = rand(model, 100_000)

            approx_pdf = kernel(data[:,2])
            true_dens = pdf(approx_pdf, rts)

            @test dens ≈ true_dens rtol = .05
            @test maximum(abs.(true_dens .- dens)) < .07
        end

        @safetestset "pdf_rt 2" begin 
            using Test 
            using Distributions
            using Random
            using SequentialSamplingModels
            using SequentialSamplingModels: pdf_rt 
            using Statistics 
            include("KDE.jl")
    
            Random.seed!(6541)
    
            model = CDDM(;
                ν=[1.75,2.0],
                η = [.50,.50],
                σ = .50,
                α = 1.0,
                τ = .30)

            rts = range(model.τ, 1.5, length=200)
            dens = map(rt -> pdf_rt(model, rt), rts)
            data = rand(model, 100_000)

            approx_pdf = kernel(data[:,2])
            true_dens = pdf(approx_pdf, rts)

            @test dens ≈ true_dens rtol = .05
            @test maximum(abs.(true_dens .- dens)) < .15
        end
    end

    @safetestset "pdf_angle" begin 
        @safetestset "pdf_angle 1" begin 
            using Test 
            using Distributions
            using Random
            using SequentialSamplingModels
            using SequentialSamplingModels: pdf_angle 
            using Statistics 
            include("KDE.jl")
    
            Random.seed!(4556)
    
            model = CDDM(;
                ν=[1.75,1.0],
                η = [.50,.50],
                σ = .50,
                α = 2.5,
                τ = .20)

            θs = range(-π, π, length=200)
            dens = map(θ -> pdf_angle(model, θ), θs)
            data = rand(model, 100_000)

            approx_pdf = kernel(data[:,1])
            true_dens = pdf(approx_pdf, θs)

            @test dens ≈ true_dens rtol = .05
            @test maximum(abs.(true_dens .- dens)) < .07
        end

        @safetestset "pdf_angle 2" begin 
            using Test 
            using Distributions
            using Random
            using SequentialSamplingModels
            using SequentialSamplingModels: pdf_angle 
            using Statistics 
            include("KDE.jl")
    
            Random.seed!(6541)
    
            model = CDDM(;
                ν=[1.75,2.0],
                η = [.50,.50],
                σ = .50,
                α = 1.0,
                τ = .30)

            θs = range(-π, π, length=200)
            dens = map(θ -> pdf_angle(model, θ), θs)
            data = rand(model, 100_000)

            approx_pdf = kernel(data[:,1])
            true_dens = pdf(approx_pdf, θs)

            @test dens ≈ true_dens rtol = .05
            @test maximum(abs.(true_dens .- dens)) < .07
        end
    end
end

