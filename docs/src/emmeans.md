# Estimated Marginal a.k.a. Least Square Means

At their simplest, estimated marginal means, a.k.a least-square means, are just effects: predictions and associated errors computed from a model at specific points on a reference grid.
The "marginal" here refers to estimation at the margins of the table, i.e. either averaging over or computing at specific values of other variables, often the means of other variables.
Effects.jl provides a convenient interface for computing EM means at all levels of categorical variables and the means of any continuous covariates. An example will make this clear:

```@example emmeans
using DataFrames, Effects, GLM, StatsModels, StableRNGs, StandardizedPredictors
rng = StableRNG(42)
growthdata = DataFrame(; age=[13:20; 13:20],
                       sex=repeat(["male", "female"], inner=8),
                       weight=[range(100, 155; length=8); range(100, 125; length=8)] .+ randn(rng, 16))
model_scaled = lm(@formula(weight ~ 1 + sex * age), growthdata;
                  contrasts=Dict(:age => ZScore(), :sex => DummyCoding()))

emmeans(model_scaled)
```

(This is the same example data as [Interaction Terms in Effects](@ref).)

Notice that we could have achieved the same result by using `effects` and specifying the levels manually:

```@example emmeans
design = Dict(:sex => ["female", "male"], :age => 16.5)
grid = expand_grid(design)
effects!(grid, model_scaled)
```

`emmeans` is primarily a convenience function for computing calling effects at a convenient, pre-defined reference grid.
Notably, `emmeans` includes information by default about all variables, even the ones analyzed
only at their typical values.
This differs from `effects!`:

```@example emmeans
design = Dict(:sex => ["female", "male"])
grid = expand_grid(design)
effects!(grid, model_scaled)
```

## Pairwise Comparisons

EM means are also useful for post-hoc examination of pairwise differences.
This is implemented via the [`empairs`](@ref) function:

```@example emmeans
empairs(model_scaled)
```

Or for a more advanced case (the same as in [The Effect of Contrast Coding](@ref)):

```@example emmeans
rng = StableRNG(42)
reps = 5
sd = 50
wtdat = DataFrame(; feed=repeat(["A", "B", "C"], inner=reps),
                  weight=[180 .+ sd * randn(rng, reps);
                          220 .+ sd * randn(rng, reps);
                          300 .+ sd * randn(rng, reps)])
model_feed = lm(@formula(weight ~ 1 + feed), wtdat)
empairs(model_feed)
```

## A Final Note

There are many subleties in more advanced applications of EM means.
These are discussed at length in the documentation for the R package `emmeans`.
For example:

- [the impact of transformations and link functions](https://cran.r-project.org/web/packages/emmeans/vignettes/transformations.html)
- [derived covariates](https://cran.r-project.org/web/packages/emmeans/vignettes/basics.html#depcovs),  e.g., including both `x` and `x^2` in a model
- [weighting of cells in computing means](https://cran.r-project.org/web/packages/emmeans/vignettes/basics.html#weights)

All of these problems are fundamental to EM means as a technique and not particular to the software implementation.
Effects.jl does not yet support everything that R's `emmeans` does, but all of the fundamental statistical concerns discussed in the latter's documentation still apply.
