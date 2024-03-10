@safetestset "sample_subject_parms" begin
    using DataFrames 
    using Distributions
    using Random
    using SequentialSamplingModels
    using Test 

    Random.seed!(684)
    n_subjects = 10_000
    Σ = [.2 0; 0 .2]

    dists = (
        ν = MvNormal([2,1], Σ),
        A = truncated(Normal(.8, .2), 0, Inf),
        k = truncated(Normal(.2, .1), 0, Inf),
        τ = truncated(Normal(.2, .1), 0, Inf)
    )
    
    subject_parms = sample_subject_parms(LBA, dists, n_subjects)

    @test size(subject_parms) == (n_subjects,)
    @test keys(subject_parms[1]) == (:ν, :A, :k, :τ)

    #flat_parms = [subject_parms[s][c] for c ∈ CartesianIndices(group_dists[1]) for s ∈ 1:n_subjects]
    df = DataFrame(subject_parms)
    #df.id = repeat(1:n_subjects, inner=4)
    #df.condition = repeat(1:4, outer = n_subjects)
    
    @test mean(df.A) ≈ mean(truncated(Normal(.8, .2), 0, Inf)) atol = 1e-2
    @test std(df.A) ≈ std(truncated(Normal(.8, .2), 0, Inf)) atol = 1e-2
    @test mean(df.k) ≈ mean(truncated(Normal(.2, .1), 0, Inf)) atol = 1e-2
    @test std(df.k) ≈ std(truncated(Normal(.2, .1), 0, Inf)) atol = 1e-2
    @test mean(df.τ) ≈ mean(truncated(Normal(.2, .1), 0, Inf)) atol = 1e-2
    @test std(df.τ) ≈ std(truncated(Normal(.2, .1), 0, Inf)) atol = 1e-2
    νs = stack(df.ν)'
    @test mean(νs, dims=1)[:] ≈ mean(MvNormal([2,1], Σ)) atol = 1e-2
    @test cov(νs) ≈ cov(MvNormal([2,1], Σ)) atol = 1e-2

    
    # df_ν = transform(df, :ν => AsTable)
    # combine(groupby(df_ν, :condition), :x1 => mean)

end
    
@safetestset "sample_subject_parms condition" begin
    using DataFrames 
    using Distributions
    using Random
    using SequentialSamplingModels
    using Test 

    Random.seed!(684)
    n_subjects = 1000
    Σ = [.2 0; 0 .2]
    
    dists = (
        A = truncated(Normal(.8, .2), 0, Inf),
        k = truncated(Normal(.2, .1), 0, Inf),
        τ = truncated(Normal(.2, .1), 0, Inf)
    )
    
    cond_dists = (
        ν = [            
            MvNormal([1,1], Σ) MvNormal([3,1], Σ) 
            MvNormal([2,1], Σ) MvNormal([4,1], Σ)
        ],
    )
    
    subject_parms = sample_subject_parms(LBA, dists, cond_dists, n_subjects)

    @test size(subject_parms) == (n_subjects,)
    @test size(subject_parms[1]) == (2,2)
    @test keys(subject_parms[1][1]) == (:ν, :A, :k, :τ)

    flat_parms = [subject_parms[s][c] for s ∈ 1:n_subjects for c ∈ CartesianIndices(cond_dists[1]) ]
    df = DataFrame(flat_parms)
    df.id = repeat(1:n_subjects, inner=4)
    df.condition = repeat(1:4, outer = n_subjects)
    
    @test mean(df.A) ≈ mean(truncated(Normal(.8, .2), 0, Inf)) atol = 1e-2
    @test std(df.A) ≈ std(truncated(Normal(.8, .2), 0, Inf)) atol = 1e-2
    @test mean(df.k) ≈ mean(truncated(Normal(.2, .1), 0, Inf)) atol = 1e-2
    @test std(df.k) ≈ std(truncated(Normal(.2, .1), 0, Inf)) atol = 1e-2
    @test mean(df.τ) ≈ mean(truncated(Normal(.2, .1), 0, Inf)) atol = 1e-2
    @test std(df.τ) ≈ std(truncated(Normal(.2, .1), 0, Inf)) atol = 1e-2
    
    df_ν = transform(df, :ν => AsTable)
    combine(groupby(df_ν, :condition), :x1 => mean)

end