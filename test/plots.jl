@safetestset "run plots" begin
    @safetestset "DDM" begin
        using Plots 
        using SequentialSamplingModels
        using Test
    
        dist = DDM()

        h = histogram(dist)
        plot!(h, dist)
        
        histogram(dist)
        plot!(dist)
        
        plot(dist)
        histogram!(dist)
        
        p = plot(dist)
        histogram!(p, dist)

        plot_model(dist; 
            n_sim=2, 
            add_density=true, 
            density_kwargs=(;t_range=range(.3, 1, length=200),))

    end

    @safetestset "RDM" begin
        using Plots 
        using SequentialSamplingModels
        using Test

        dist = RDM(;ν=[1,2,3], k=.30, A=.70, τ=.20)
        h = histogram(dist)
        plot!(h, dist)
        
        histogram(dist)
        plot!(dist)
        
        plot(dist)
        histogram!(dist)
        
        p = plot(dist)
        histogram!(p, dist)

        dist = RDM()
        density_kwargs=(;t_range=range(.20, 0.80, length=100),)
        plot_model(dist; n_sim=10, add_density=true, density_kwargs, xlims=(0,0.80))
        
    end
  
    @safetestset "LCA" begin
        using Plots 
        using SequentialSamplingModels
        using Test

        dist = LCA()
        p = plot(dist; t_range=range(.3, 1.2, length=100))
        histogram!(p, dist)
        
        plot(dist; t_range=range(.3, 1.2, length=100))
        histogram!(dist)
        
        h = histogram(dist)
        plot!(h, dist; t_range=range(.3, 1.2, length=100))
        
        histogram(dist)
        plot!(dist; t_range=range(.3, 1.2, length=100))

        density_kwargs=(;t_range=range(.20, 1.50, length=100),)
        plot_model(dist; n_sim=10, add_density=true, density_kwargs, xlims=(0,1.50))
    end

    @safetestset "Wald" begin
        using Plots 
        using SequentialSamplingModels
        using Test

        dist = Wald(ν=3.0, α=.5, τ=.130)
        h = histogram(dist)
        plot!(h, dist; t_range=range(.130, 1.0, length=100))
        
        histogram(dist)
        plot!(dist; t_range=range(.130, 1.0, length=100))
        
        p = plot(dist; t_range=range(.130, 1.0, length=100))
        histogram!(p, dist)
        
        plot(dist; t_range=range(.130, 1.0, length=100))
        histogram!(dist)


        dist = Wald()
        density_kwargs=(;t_range=range(.2,1.2, length=100),)
        plot_model(dist; add_density=true, n_sim=2, density_kwargs, xlims=(0,1.2))
    end

    @safetestset "CDDM" begin
        using Plots 
        using SequentialSamplingModels
        using Test

        dist = CDDM(;ν=[1.5,1.5], η=[1,1], σ=1, α=2.5, τ=0.30)
        h = histogram(dist)
        plot!(h, dist)
        
        histogram(dist)
        plot!(dist)
        
        p = plot(dist)
        histogram!(p, dist)
        
        plot(dist)
        histogram!(dist)

        dist = CDDM(;ν=[1.5,1.5], η=[1,1], σ=1, α=2.5, τ=0.30)
        plot_model(dist; lims=(-5,5))    
    end

    @safetestset "LBA" begin
        using Plots 
        using SequentialSamplingModels
        using Test

        dist = LBA()

        h = histogram(dist)
        plot!(h, dist)
        
        histogram(dist)
        plot!(dist)
        
        p = plot(dist)
        histogram!(p, dist)
        
        plot(dist)
        histogram!(dist)

        density_kwargs=(;t_range=range(.3,1.2, length=100),)
        plot_model(dist; add_density=true, n_sim=2, density_kwargs, xlims=(0,1.2))

    end
    
    @safetestset "WaldMixture" begin
        using Plots 
        using SequentialSamplingModels
        using Test

        dist = WaldMixture()

        h = histogram(dist)
        plot!(h, dist)
        
        histogram(dist)
        plot!(dist)
        
        p = plot(dist)
        histogram!(p, dist)
        
        plot(dist)
        histogram!(dist)

        density_kwargs=(;t_range=range(.13,.60, length=100),)
        plot_model(dist; add_density=true, n_sim=2, density_kwargs, xlims=(0,.60))
    end

    @safetestset "PoissonRace" begin
        using Plots 
        using SequentialSamplingModels
        using Test

        dist = PoissonRace(;ν=[.04,.045], α=[4,3], τ=0.20)
        h = histogram(dist)
        plot!(h, dist)
        
        histogram(dist)
        plot!(dist)
        
        plot(dist)
        histogram!(dist)
        
        p = plot(dist)
        histogram!(p, dist)

        plot_model(dist; lims=(0,1))
        @test true
    end
end

@safetestset "plot_quantiles" begin 
    @safetestset "1D" begin 
        using Distributions
        using Plots 
        using SequentialSamplingModels
        using Test 

        dist = Normal(0, 1)
        x = rand(dist, 100)
        preds = [rand(dist, 100) for _ ∈ 1:1000]

        q_preds = compute_quantiles.(preds)
        q_data = compute_quantiles(x) 
        plot_quantiles(q_data, q_preds)
        @test true
    end

    @safetestset "2D" begin 
        using Distributions
        using Plots 
        using SequentialSamplingModels
        using Test 

        dist = Normal(0, 1)
        x = rand(dist, 100)
        f(dist) = [compute_quantiles(rand(dist, 100)), compute_quantiles(rand(dist, 100))]
        q_preds = [f(dist) for _ ∈ 1:1000, _ ∈ 1:1]

        q_data = [compute_quantiles(x), compute_quantiles(x)] 
        plot_quantiles(q_data, q_preds)
        @test true
    end
end

@safetestset "plot_choices" begin 
    using Distributions
    using Plots 
    using SequentialSamplingModels
    using Test 

    θ = fill(10, 3)
    preds = [rand(Dirichlet( θ)) for _ ∈ 1:1000]

    data = rand(Dirichlet(θ)) 
    plot_choices(data, preds)
    @test true
end