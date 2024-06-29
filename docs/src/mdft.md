```@setup MDFT
using SequentialSamplingModels
using Plots 
using Random
M = [
    1.0 3.0 # A 
    3.0 1.0 # B
    0.9 3.1 # S
]
```

# Multi-attribute Decision Field Theory

Multi-attribute Decision Field Theory (MDFT; Roe, Busemeyer, & James, 2001) models how people choose between alternatives with multiple dimensions, such as cars, phones, or jobs. As an example, jobs may differ in terms of benefits, salary, flexibility, and life-work balance. As with other sequential sampling models, MDFT assumes that evidence (or preference) accumulates dynamically until the evidence for one alternative reaches a threshold, and triggers the selection of the winning alternative. In addition, MDFT is based on three other core assumptions:

1. Attention switches between attributes, and alternatives are compared on the currently attended attribute
2. As two alternatives become closer to each other in attribute space, their mutual inhibition increases
3. Evidence for each alternative gradually decays across time

One of MDFT's strong suits is accounting for context effects in preferential decision making. A context effect occurs when the preference relationship between two alternatives changes when a third alternative is included in the choice set. In such cases, the preferences may reverse or the decision maker may violate rational choice principles.  

Note that this version of MDFT uses stochastic differential equations (see Evans et al., 2019). For the random walk version, see `ClassicMDFT`. 

# Similarity Effect

In what follows, we will illustrate the use of MDFT with a demonstration of the similarity effect. 
Consider the choice between two jobs, $A$ and $B$. The main criteria for evaluating the two jobs are salary and flexibility. Job A is high on salary but low on flexibility, whereas job B is low on salary. In the plot below, jobs A and B are located on the line of indifference, $y = 3 - x$. However, because salary recieves more attention, job A is slightly prefered over job B.

```@example MDFT
scatter(
    M[:, 1],
    M[:, 2],
    grid = false,
    leg = false,
    lims = (0, 4),
    xlabel = "Flexibility",
    ylabel = "Salary",
    markersize = 6,
    markerstrokewidth = 2
)
annotate!(M[1, 1] + 0.10, M[1, 2] + 0.25, "A")
annotate!(M[2, 1] + 0.10, M[2, 2] + 0.25, "B")
annotate!(M[3, 1] + 0.10, M[3, 2] + 0.25, "S")
plot!(0:0.1:4, 4:-0.1:0, color = :black, linestyle = :dash)
```
Suppose an job `S`, which is similar to A is added to the set of alternatives. Job `S` inhibits job `A` more than job `B` because `S` and `A` are close in attribute space. As a result, the preference for job `A` over job `B` is reversed. Formally, this is stated as:  

```math
\Pr(A \mid \{A,B\}) > \Pr(B \mid \{A,B\})
```

```math
\Pr(A \mid \{A,B,S\}) < \Pr(B \mid \{A,B,S\})
```

## Load Packages
The first step is to load the required packages.

```@example MDFT
using SequentialSamplingModels
using Plots 
using Random

Random.seed!(8741)
```
## Create Model Object
In the code below, we will define parameters for the LBA and create a model object to store the parameter values. 

### Drift Rate Scalar
In MDFT, the drift rate is determined by the contrast between alternatives along the attended attribute. These evaluations are scaled by the parameter $\gamma$:

```@example MDFT
γ = 1.0
```
### Threshold
The threshold $\alpha$ represents the amount of evidence required to make a decision.
```@example MDFT 
α = .50
```

### Dominance Weight
In MDFT, alternatives are compared along the dominance dimension (diagonal) and indifference dimension (off-diagonal) in attribute space. The relative weight of the dominance dimension is controlled by parameter $\beta$
```@example MDFT 
β = 10
```

### Lateral Inhibition

In MDFT, alternatives inhibit each other as an inverse function of thier distance in attribute space: the closer they are, the more inhibititory the relationship. Lateral inhibition is controled via a
alternative $\times$ alternative feedback matrix in which the diagonal elements (e.g., self-inhibition) represents decay or leakage, and the non-diagonal elements represent lateral inhibition between different alternatives. The values of the feedback matrix are controled by a Gaussian distance function with two parameters: $\phi_1$ and $\phi_2$. 

#### Inhibition Strength

The distance gradient parameter $\phi_1$ controls the strength of lateral inhibition between alternatives:

```@example MDFT
ϕ1 = .01
```

#### Maximum Inhibition

Maximimum inhibition and decay is controlled by parameter $\phi_2$:

```@example MDFT
ϕ2 = .10
```

### Diffusion Noise
Diffusion noise is the amount of within trial noise in the evidence accumulation process. 
```@example MDFT 
σ = .10
```

### Non-Decision Time

Non-decision time is an additive constant representing encoding and motor response time. 
```@example MDFT 
τ = 0.30
```
### Attention Switching Rates
The rate at which attention shifts from one attribute to the other is controled by the following rate parameters:
```@example MDFT 
κ = [6, 5]
```
The second rate is lower than the first rate to reflect more attention to the second dimension (i.e., salary).

### MDFT Constructor 

Now that values have been asigned to the parameters, we will pass them to `MDFT` to generate the model object. We will begin with the choice between job `A` and job `B`.

```@example MDFT 
dist = MDFT(;
    n_alternatives = 2,
    σ,
    α,
    τ,
    γ,
    κ,
    ϕ1,
    ϕ2,
    β,
)
```
## Simulate Model

Now that the model is defined, we will generate 10,000 choices and reaction times using `rand`. 

 ```@example MDFT
M₂ = [
    1.0 3.0 # A 
    3.0 1.0 # B
]
    
choices,rts = rand(dist, 10_000, M₂; Δt = .001)
probs2 = map(c -> mean(choices .== c), 1:2)
```
Here, we see that job `A` is prefered over job `B`. In the code block above, `rand` has a keyword argument `Δt` which controls the precision of the discrete approximation. The default value is `Δt = .001`.

Next, we will simulate the choice between jobs `A`, `B`, and `S`.

 ```@example MDFT
dist = MDFT(;
    n_alternatives = 3,
    σ,
    α,
    τ,
    γ,
    κ,
    ϕ1,
    ϕ2,
    β,
)

M₃ = [
    1.0 3.0 # A 
    3.0 1.0 # B
    0.9 3.1 # S
]

choices,rts = rand(dist, 10_000, M₃)
probs3 = map(c -> mean(choices .== c), 1:3)
```
In this case, the preferences have reversed: job `B` is now prefered over job `A`. 

## Compute Choice Probability
The choice probability $\Pr(C=c)$ is computed by passing the model and choice index to `cdf` along with a large value for time as the second argument.

 ```@example MDFT 
cdf(dist, 1, Inf, M₃)
```

## Plot Simulation
The code below plots a histogram for each alternative.
 ```@example MDFT 
histogram(dist; model_args = (M₃,))
```
# References

Evans, N. J., Holmes, W. R., & Trueblood, J. S. (2019). Response-time data provide critical constraints on dynamic models of multi-alternative, multi-attribute choice. Psychonomic Bulletin & Review, 26, 901-933.

Hotaling, J. M., Busemeyer, J. R., & Li, J. (2010). Theoretical developments in decision
field theory: Comment on tsetsos, usher, and chater (2010). Psychological Review, 117 , 1294-1298.

Roe, Robert M., Jermone R. Busemeyer, and James T. Townsend. "Multi-attribute Decision Field Theory: A dynamic connectionst model of decision making." Psychological review 108.2 (2001): 370.