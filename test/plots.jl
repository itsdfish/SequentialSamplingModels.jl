@safetestset "run plots" begin
    @safetestset "aDDM" begin
        using Plots
        using SequentialSamplingModels
        using StatsBase
        using Test

        mutable struct Transition
            state::Int
            n::Int
            mat::Array{Float64, 2}
        end

        function Transition(mat)
            n = size(mat, 1)
            state = rand(1:n)
            return Transition(state, n, mat)
        end

        function fixate(transition)
            (; mat, n, state) = transition
            w = @view mat[state, :]
            next_state = sample(1:n, Weights(w))
            transition.state = next_state
            return next_state
        end

        tmat = Transition([
            0.98 0.015 0.005
            0.015 0.98 0.005
            0.45 0.45 0.1
        ])

        Base.broadcastable(x::Transition) = Ref(x)

        dist = aDDM()

        h = histogram(dist; model_kwargs = (; fixate), model_args = (; tmat))
        plot!(h, dist; model_kwargs = (; fixate), model_args = (; tmat))

        histogram(dist; model_kwargs = (; fixate), model_args = (; tmat))
        plot!(dist; model_kwargs = (; fixate), model_args = (; tmat))

        plot(dist; model_kwargs = (; fixate), model_args = (; tmat))
        histogram!(dist; model_kwargs = (; fixate), model_args = (; tmat))

        p = plot(dist; model_kwargs = (; fixate), model_args = (; tmat))
        histogram!(p, dist; model_kwargs = (; fixate), model_args = (; tmat))

        plot_model(
            dist;
            n_sim = 2,
            add_density = true,
            model_kwargs = (; fixate),
            model_args = (; tmat),
            density_kwargs = (; t_range = range(0.0, 4, length = 200),)
        )
    end

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

        plot_model(
            dist;
            n_sim = 2,
            add_density = true,
            density_kwargs = (; t_range = range(0.3, 1, length = 200),)
        )
    end

    @safetestset "RDM" begin
        using Plots
        using SequentialSamplingModels
        using Test

        dist = RDM(; ν = [1, 2, 3], k = 0.30, A = 0.70, τ = 0.20)
        h = histogram(dist)
        plot!(h, dist)

        histogram(dist)
        plot!(dist)

        plot(dist)
        histogram!(dist)

        p = plot(dist)
        histogram!(p, dist)

        dist = RDM()
        density_kwargs = (; t_range = range(0.20, 0.80, length = 100),)
        plot_model(dist; n_sim = 10, add_density = true, density_kwargs, xlims = (0, 0.80))
    end

    @safetestset "LCA" begin
        using Plots
        using SequentialSamplingModels
        using Test

        dist = LCA()
        p = plot(dist; t_range = range(0.3, 1.2, length = 100))
        histogram!(p, dist)

        plot(dist; t_range = range(0.3, 1.2, length = 100))
        histogram!(dist)

        h = histogram(dist)
        plot!(h, dist; t_range = range(0.3, 1.2, length = 100))

        histogram(dist)
        plot!(dist; t_range = range(0.3, 1.2, length = 100))

        density_kwargs = (; t_range = range(0.20, 1.50, length = 100),)
        plot_model(dist; n_sim = 10, add_density = true, density_kwargs, xlims = (0, 1.50))
    end

    @safetestset "MDFT" begin
        using Plots
        using SequentialSamplingModels
        using Test

        parms = (
            σ = 0.1,
            α = 0.50,
            τ = 0.0,
            γ = 1.0,
            ϕ1 = 0.01,
            ϕ2 = 0.1,
            β = 10,
            κ = [5, 5]
        )

        dist = MDFT(; n_alternatives = 3, parms...)

        M = [
            1.0 3.0
            3.0 1.0
            0.9 3.1
        ]

        h = histogram(dist; model_args = (M,))
        plot!(h, dist; model_args = (M,))

        histogram(dist; model_args = (M,))
        plot!(dist; model_args = (M,))

        plot(dist; model_args = (M,))
        histogram!(dist; model_args = (M,))

        p = plot(dist; model_args = (M,))
        histogram!(p, dist; model_args = (M,))

        plot_model(
            dist;
            n_sim = 2,
            add_density = true,
            model_args = (M,),
            density_kwargs = (; t_range = range(0.1, 1, length = 200),)
        )
    end

    @safetestset "Wald" begin
        using Plots
        using SequentialSamplingModels
        using Test

        dist = Wald(ν = 3.0, α = 0.5, τ = 0.130)
        h = histogram(dist)
        plot!(h, dist; t_range = range(0.130, 1.0, length = 100))

        histogram(dist)
        plot!(dist; t_range = range(0.130, 1.0, length = 100))

        p = plot(dist; t_range = range(0.130, 1.0, length = 100))
        histogram!(p, dist)

        plot(dist; t_range = range(0.130, 1.0, length = 100))
        histogram!(dist)

        dist = Wald()
        density_kwargs = (; t_range = range(0.2, 1.2, length = 100),)
        plot_model(dist; add_density = true, n_sim = 2, density_kwargs, xlims = (0, 1.2))
    end

    @safetestset "CDDM" begin
        using Plots
        using SequentialSamplingModels
        using Test

        dist = CDDM(; ν = [1.5, 1.5], η = [1, 1], σ = 1, α = 2.5, τ = 0.30)
        h = histogram(dist)
        plot!(h, dist)

        histogram(dist)
        plot!(dist)

        p = plot(dist)
        histogram!(p, dist)

        plot(dist)
        histogram!(dist)

        dist = CDDM(; ν = [1.5, 1.5], η = [1, 1], σ = 1, α = 2.5, τ = 0.30)
        plot_model(dist; lims = (-5, 5))
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

        density_kwargs = (; t_range = range(0.3, 1.2, length = 100),)
        plot_model(dist; add_density = true, n_sim = 2, density_kwargs, xlims = (0, 1.2))
    end

    @safetestset "LNR" begin
        using Plots
        using SequentialSamplingModels
        using Test

        dist = LNR()

        h = histogram(dist)
        plot!(h, dist)

        histogram(dist)
        plot!(dist)

        p = plot(dist)
        histogram!(p, dist)

        plot(dist)
        histogram!(dist)

        density_kwargs = (; t_range = range(0.1, 1.2, length = 100),)
        plot_model(dist; add_density = true, n_sim = 2, density_kwargs, xlims = (0, 1.2))
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

        density_kwargs = (; t_range = range(0.13, 0.60, length = 100),)
        plot_model(dist; add_density = true, n_sim = 2, density_kwargs, xlims = (0, 0.60))
    end

    @safetestset "PoissonRace" begin
        using Plots
        using SequentialSamplingModels
        using Test

        dist = PoissonRace(; ν = [0.04, 0.045], α = [4, 3], τ = 0.20)
        h = histogram(dist)
        plot!(h, dist)

        histogram(dist)
        plot!(dist)

        plot(dist)
        histogram!(dist)

        p = plot(dist)
        histogram!(p, dist)

        plot_model(dist; lims = (0, 1))
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
    preds = [rand(Dirichlet(θ)) for _ ∈ 1:1000]

    data = rand(Dirichlet(θ))
    plot_choices(data, preds)
    @test true
end
