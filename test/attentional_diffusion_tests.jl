@safetestset "Attentional Diffusion" begin
    @safetestset "aDDM" begin 
        using Test, SequentialSamplingModels, StatsBase, Random
        Random.seed!(5511)
       
        mutable struct Transition
            state::Int 
            n::Int
            mat::Array{Float64,2} 
        end
    
        function Transition(mat)
            n = size(mat,1)
            state = rand(1:n)
            return Transition(state, n, mat)
        end
        
        function attend(t)
            w = t.mat[t.state,:]
            next_state = sample(1:t.n, Weights(w))
            t.state = next_state
            return next_state
        end
    
        model = aDDM(;ν1=5.0, ν2=4.0)
    
        tmat = Transition([.98 .015 .005;
                            .015 .98 .005;
                            .45 .45 .1])
    
        rts1 = rand(model, 1000, x -> attend(x), tmat)
        n1,n2 = length.(rts1)
        @test n1 > n2
       
        model = aDDM(;ν1=4.0, ν2=5.0)
        rts2 = rand(model, 1000, x -> attend(x), tmat)
        n1,n2 = length.(rts2)
        @test n1 < n2
    
        μ_rts1 = mean(vcat(rts1...))
        model = aDDM(;ν1=5.0, ν2=5.0)
        rts3 = rand(model, 1000, x -> attend(x), tmat)
        μ_rts3 = mean(vcat(rts3...))
        @test μ_rts1 < μ_rts3
    end
end