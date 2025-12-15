function test_context_effect(rng, parms, rand_parms, M; test_func, n_sim = 1000)
    probs2, probs3 = simulate_context_effect(rng, parms, rand_parms, M; n_sim)
    return test_func(probs2, probs3)
end

function simulate_context_effect(rng, parms, rand_parms, M; n_sim = 1000)
    n_alternatives = 2
    model = MDFT(; n_alternatives, parms..., rand_parms...)
    M2 = M[1:n_alternatives, :]
    choices, _ = rand(rng, model, n_sim, M2)
    probs2 = map(c -> mean(choices .== c), 1:n_alternatives)

    n_alternatives = 3
    model = MDFT(; n_alternatives, parms..., rand_parms...)
    choices, _ = rand(rng, model, n_sim, M)
    probs3 = map(c -> mean(choices .== c), 1:n_alternatives)
    return probs2, probs3
end

test_attraction(probs2, probs3) = (probs3[1] - probs2[2]) ≥ 0.005
test_compromise(probs2, probs3) =
    isapprox(probs2..., atol = 0.02) && ((probs3[2] - probs3[1]) ≥ 0.005) &&
    ((probs3[2] - probs3[3]) ≥ 0.005)
test_similarity(probs2, probs3) =
    ((probs2[1] - probs2[2]) ≥ 0.005) && ((probs3[2] - probs3[1]) ≥ 0.005)

function run_tests(rng, parms; Ma, Ms, Mc)
    rand_parms = (
        ϕ1 = round(rand(rng, Uniform(0.5, 1)), digits = 2),
        ϕ2 = round(rand(rng, Uniform(0, 2)), digits = 2),
        β = round(rand(rng, Uniform(5, 15)), digits = 2),
        κ = round.(fill(rand(rng, Uniform(1, 20)), 2), digits = 2)
    )
    attraction =
        test_context_effect(rng, parms, rand_parms, Ma; test_func = test_attraction)
    compromise =
        test_context_effect(rng, parms, rand_parms, Mc; test_func = test_compromise)
    similarity =
        test_context_effect(rng, parms, rand_parms, Ms; test_func = test_similarity)
    return (; attraction, compromise, similarity, rand_parms...)
end

# parms = (
#     σ = 0.1,
#     α = 0.50,
#     τ = 0.0,
#     γ = 1.0,
#     κ = [5.0, 5.0],
#     ϕ1 = 0.01,
#     ϕ2 = 5,
#     β = 10.0
# )
# # stimuli for attraction effect
# Ma = [
#     3.0 1.0
#     1.0 3.0
#     0.50 2.5
# ]

# # stimuli for compromise effect
# Mc = [
#     1.0 3.0
#     2.0 2.0
#     3.0 1.0
# ]

# # stimuli for similarity effect
# Ms = [
#     1.0 3.0
#     3.0 1.0
#     0.9 3.1
# ]

# results = map(i -> begin
#     output = run_tests(parms; Ma, Ms, Mc)
#     println("i $i $output")
#     return output
# end, 1:100)

# filter(x -> x.attraction && x.compromise && x.similarity, results)
