```@setup MLBA
using SequentialSamplingModels
using Plots 
using Random
M = [
    1.0 3.0 # A 
    3.0 1.0 # B
    0.9 3.1 # S
]
```

# Multi-attribute Linear Ballistic Accumulator

Multi-attribute Linear Ballistic Accumulator (MLBA; Trueblood et al., 2014) is an extention of the [LBA](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/lba/) for multi-attribute deicisions. Alternatives in multi-attribute decisions vary along multiple dimensions. For example, jobs may differ in terms of benefits, salary, flexibility, and work-life balance. As with other sequential sampling models, MLBA assumes that evidence (or preference) accumulates dynamically until the evidence for one alternative reaches a threshold, and triggers the selection of the winning alternative. MLBA incorporates three additional core assumptions:

1. Drift rates based on a weighted sum of pairwise comparisons
2. The comparison weights are an inverse function of similarity of two alternatives on a given attribute.
3. Objective values are mapped to subjective values, which can display extremeness aversion

As with [MDFT](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/mdft/), the MLBA excells in  accounting for context effects in preferential decision making. A context effect occurs when the preference relationship between two alternatives changes when a third alternative is included in the choice set. In such cases, the preferences may reverse or the decision maker may violate rational choice principles.  

# Similarity Effect

In what follows, we will illustrate the use of MLBA with a demonstration of the similarity effect. 
Consider the choice between two jobs, `A` and `B`. The main criteria for evaluating the two jobs are salary and flexibility. Job `A` is high on salary but low on flexibility, whereas job `B` is low on salary. In the plot below, jobs `A` and `B` are located on the line of indifference, $y = 3 - x$. However, because salary recieves more attention, job `A` is slightly prefered over job `B`.

```@example MLBA
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

```@example MLBA
using SequentialSamplingModels
using Plots 
using Random

Random.seed!(8741)
```
## Create Model Object
In the code below, we will define parameters for the MLBA and create a model object to store the parameter values. 

### Drift Rate Parameters
In sequential sampling models, the drift rate is the average speed with which evidence accumulates towards a decision threshold. In the MLBA, the drift rate is determined by comparing attributes between alternatives. The drift rate for alternative $i$ is defined as:

$\nu_i = \beta_0 + \sum_{i\ne j} v_{i,j},$

where $\beta_0$ is the basline additive constant, and $v_{i,j}$ is the comparative value between alternatives $i$ and $j$. A given alternative $i$ is compared to another alternative $j \ne i$ as a weighted sum of differences across attributes $k \in [1,2,\dots n_a]$:

$v_{i,j} = \sum_{k=1}^{n_a} w_{i,j,k} (u_{i,k} - u_{j,k}).$
The attention weight between alternatives $i$ and $j$ on attribute $k$ is an inverse function of similarity, and decays exponentially with distance:

$w_{i,j,k} = e^{-\lambda |u_{i,k} - u_{j,k}|}$

Similarity between alternatives is not necessarily symmetrical, giving rise to two decay rates:
```math
\lambda = \begin{cases} 
      \lambda_p & u_{i,k} \geq u_{j,k}\\
      \lambda_n & \mathrm{ otherwise}\\
\end{cases}
```
The subjective value $\mathbf{u} = [u_1,u_2]$ is found by bending the line of indifference passing through the objective stimulus $\mathbf{s}$ in attribute space, such that $(\frac{x}{a})^\gamma + (\frac{x}{b})^\gamma = 1$. When $\gamma > 1$, the model produces extremeness aversion. 
#### Baseline Drift Rate

The baseline drift rate parameter $\beta_0$ is a constant added to drift rate:

```@example MLBA
β₀ = 5.0
```

#### Similarity-Based Weighting

Attention weights are an inverse function of similarity between alternatives on a given attribute. The decay rate for positive differences is:

```@example MLBA
λₚ = 0.20
```
The decay rate for negative differences is:
```@example MLBA
λₙ = 0.40
```
The comparisons are asymmetrical when $\lambda_p \ne \lambda_n$. 

### Extremeness Aversion 
The parameter for extremeness aversion is: 
```@example MLBA
γ = 5
```

 $\gamma = 1$ indicates objective treatment of stimuli, whereas $\gamma > 1$ indicates extreness aversion, i.e. $[2,2]$ is prefered to $[3,1]$ even though both fall along the line of indifference.

### Standard Deviation of Drift Rates
The standard deviation of the drift rate distribution is given by $\sigma$, which is commonly fixed to 1 for each accumulator.

```@example MLBA
σ = [1.0,1.0]
```

### Maximum Starting Point

The starting point of each accumulator is sampled uniformly between $[0,A]$.

```@example MLBA 
A = 1.0
```
### Threshold - Maximum Starting Point

Evidence accumulates until accumulator reaches a threshold $\alpha = k +A$. The threshold is parameterized this way to faciliate parameter estimation and to ensure that $A \le \alpha$.

```@example MLBA 
k = 1.0
```
### Non-Decision Time

Non-decision time is an additive constant representing encoding and motor response time. 

```@example MLBA 
τ = 0.30
```

### MLBA Constructor 

Now that values have been asigned to the parameters, we will pass them to `MLBA` to generate the model object. We will begin with the choice between job `A` and job `B`.

```@example MLBA 
dist = MLBA(;
    n_alternatives = 2,
    β₀,
    λₚ,
    λₙ,
    γ,
    τ,
    A,
    k,
)
```
## Simulate Model

Now that the model is defined, we will generate 10,000 choices and reaction times using `rand`. 

 ```@example MLBA
M₂ = [
    1.0 3.0 # A 
    3.0 1.0 # B
]
    
choices,rts = rand(dist, 10_000, M₂)
probs2 = map(c -> mean(choices .== c), 1:2)
```
Here, we see that job `A` is prefered over job `B`.

Next, we will simulate the choice between jobs `A`, `B`, and `S`.

 ```@example MLBA
dist = MLBA(;
    n_alternatives = 3,
    β₀,
    λₚ,
    λₙ,
    γ,
    τ,
    A,
    k,
)

M₃ = [
    1.0 3.0 # A 
    3.0 1.0 # B
    0.9 3.1 # S
]

choices,rts = rand(dist, 10_000, M₃)
probs3 = map(c -> mean(choices .== c), 1:3)
```
In this case, the preferences have reversed: job `B` is now preferred over job `A`. 

## Compute Choice Probability
The choice probability $\Pr(C=c)$ is computed by passing the model and choice index to `cdf` along with a large value for time as the second argument.

 ```@example MLBA 
cdf(dist, 1, 100, M₃)
```

## Plot Simulation
The code below plots a histogram for each alternative.
 ```@example MLBA 
histogram(dist; model_args = (M₃,))
```
# References

Trueblood, J. S., Brown, S. D., & Heathcote, A. (2014). The multiattribute linear ballistic accumulator model of context effects in multialternative choice. Psychological Review, 121(2), 179.
