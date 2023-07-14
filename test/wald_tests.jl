@safetestset "Wald" begin
    @safetestset "pdf" begin 
        using Test, SequentialSamplingModels, Random
        include("KDE.jl")
        Random.seed!(22158)
        d = Wald(2, 1, .1)
        @test mean(d) ≈ (1/2) + .1 atol = 1e-5


        function simulate(υ, α, τ)
            noise = 1.0
            #Time Step
            Δt = .0005
            #Evidence step
            Δe = noise * sqrt(Δt)
            e = 0.0
            t = τ
            p = .5 * (1 + υ * sqrt(Δt) / noise)
            while (e < α)
                t += Δt
                e += rand() ≤ p ? Δe : -Δe
            end
            return t
        end
        rts = map(_ -> simulate(3, 1, .2), 1:10^5)
        approx_pdf = kde(rts)
        x = .2:.01:1.5
        y′ = pdf(approx_pdf, x)
        y = pdf.(Wald(3,1,.2), x)
        @test y′ ≈ y rtol = .03
        @test mean(rts) ≈ mean(Wald(3,1,.2)) atol = 5e-3
        @test std(rts) ≈ std(Wald(3,1,.2)) atol = 1e-3
    end
    
    @safetestset "loglikelihood" begin 
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(655)

        dist = Wald(2, 1, .1)
        rt = rand(dist, 10)

        sum_logpdf = logpdf.(dist, rt) |> sum 
        loglike = loglikelihood(dist, rt)
        @test sum_logpdf ≈ loglike 
    end
end