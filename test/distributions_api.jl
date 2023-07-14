@safetestset "Distributions API" begin
    @safetestset "minimum" begin
        using SequentialSamplingModels

        models = [
            LBA(ν=[3.0,2.0], A = .8, k = .2, τ = .3),
            LCA(;α = 1.5, β=0.20, λ=0.10, ν=[2.5,2.0], Δt=.001, τ=.30, σ=1.0),
            LNR(ν=[-1.0,3.0], σ=.3, τ=0.0),
            RDM(;ν=[1.0,.5], k=0.5, A=1.0, τ=.2),
            Wald(3,1,.2),
            WaldMixture(2, .2, 1, .1),
            aDDM(),
            maaDDM()]

        for m ∈ models 
            @test minimum(m) == 0.0
        end
    end

    @safetestset "maximum" begin
        using SequentialSamplingModels

        models = [
            LBA(ν=[3.0,2.0], A = .8, k = .2, τ = .3),
            LCA(;α = 1.5, β=0.20, λ=0.10, ν=[2.5,2.0], Δt=.001, τ=.30, σ=1.0),
            LNR(ν=[-1.0,3.0], σ=.3, τ=0.0),
            RDM(;ν=[1.0,.5], k=0.5, A=1.0, τ=.2),
            Wald(3,1,.2),
            DDM(),
            WaldMixture(2, .2, 1, .1),
            aDDM(),
            maaDDM()]

        for m ∈ models 
            @test maximum(m) == Inf
        end
    end

    @safetestset "insupport" begin
        using SequentialSamplingModels
        using Distributions

        models = [
            LBA(ν=[3.0,2.0], A = .8, k = .2, τ = .3),
            LCA(;α = 1.5, β=0.20, λ=0.10, ν=[2.5,2.0], Δt=.001, τ=.30, σ=1.0),
            LNR(ν=[-1.0,3.0], σ=.3, τ=0.0),
            RDM(;ν=[1.0,.5], k=0.5, A=1.0, τ=.2),
            DDM(),
            aDDM(),
            maaDDM()]

        for m ∈ models 
            @test insupport(m, (choice=1,rt=1.0))
            @test !insupport(m, (choice=1,rt=-1.0))
        end

        models = [Wald(3,1,.2),
            WaldMixture(2, .2, 1, .1)]

        for m ∈ models 
            @test insupport(m, 1.0)
            @test !insupport(m, -1.0)
        end
    end
end
