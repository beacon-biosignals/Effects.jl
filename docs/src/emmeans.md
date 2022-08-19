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

`emmeans` is primarily a convenience function for computing effects on a convenient, pre-defined reference grid.
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

Or for a more advanced case:

```@example emmeans
using MixedModels
kb07 = MixedModels.dataset(:kb07)
mixed_form = @formula(rt_trunc ~ 1 + spkr * prec * load + (1|item) + (1|subj))
mixed_model = fit(MixedModel, mixed_form, kb07; progress=false)
empairs(mixed_model)
```

If we provide a method for computing the degrees of freedom or an appropriate value, then `empairs` will also generate test statistics:

```@example emmeans
empairs(model_scaled; dof=dof_residual)
```
These degrees of freedom correspond to the denominator degrees of freedom in the associated F-tests.

### Mixed-effects Models

For mixed-effects models, the denominator degrees of freedom are not clearly defined[^GLMMFAQ] (nor is it clear that the asymptotic distribution is even F for anything beyond a few special cases[^Bates2006]).
The fundamental problem is that mixed models have in some sense "more" degrees of freedom than you would expect from naively counting parameters -- the conditional modes (BLUPs) aren't technically parameters, but do somehow contribute to increasing the amount of wiggle-room that the model has for adapting to the data.
We can try a few different approaches and see how they differ:

- the default definition in MixedModels.jl for `dof_residual(model)` is simply `nobs(model) - dof(model)`
```@example emmeans
empairs(mixed_model; dof=dof_residual)
```
- the leverage formula, given by `nobs(model) - sum(leverage(model))` (for classical linear models, this is exactly the same as (1))
```@example emmeans
empairs(mixed_model; dof=nobs(mixed_model) - sum(leverage(mixed_model)))
```
- counting each conditional mode as a parameter (this is the most conservative value because it has the smallest degrees of freedom)
```@example emmeans
empairs(mixed_model; dof=nobs(mixed_model) - sum(size(mixed_model)[2:3]))
```
- the infinite degrees of freedom (i.e. interpret $t$ as $z$ and $F$ as $\chi^2$ )
```@example emmeans
empairs(mixed_model; dof=Inf)
```

These values are all very similar to each other, because the $t$ distribution rapidly converges to the $z$ distribution for $t > 30$ and so amount of probability mass in the tails does not change much for a model with more than a thousand observations and 30+ levels of each grouping variable.
<!-- XXX
should add documentation about using Satterthwaite and Kenward-Roger but
there's not a Julia implementation of those things yet, so not high priority
-->

### Multiple Comparisons Correction

A perhaps more important concern is inflation of Type-1 error rate through multiple testing.
We can also provide a correction method to be applied to the p-values.
```@example emmeans
using MultipleTesting
bonferroni(pvals) = adjust(PValues(pvals), Bonferroni())
empairs(mixed_model; dof=Inf, padjust=bonferroni)
```

[^GLMMFAQ]: [Why doesn’t lme4 display denominator degrees of freedom/p values? What other options do I have?](https://bbolker.github.io/mixedmodels-misc/glmmFAQ.html#why-doesnt-lme4-display-denominator-degrees-of-freedomp-values-what-other-options-do-i-have)
[^Bates2006]: [lmer, p-values and all that](https://stat.ethz.ch/pipermail/r-help/2006-May/094765.html)

## A Final Note

There are many subleties in more advanced applications of EM means.
These are discussed at length in the documentation for the R package `emmeans`.
For example:

- [the impact of transformations and link functions](https://cran.r-project.org/web/packages/emmeans/vignettes/transformations.html)
- [derived covariates](https://cran.r-project.org/web/packages/emmeans/vignettes/basics.html#depcovs), e.g., including both `x` and `x^2` in a model
- [weighting of cells in computing means](https://cran.r-project.org/web/packages/emmeans/vignettes/basics.html#weights)

All of these problems are fundamental to EM means as a technique and not particular to the software implementation.
Effects.jl does not yet support everything that R's `emmeans` does, but all of the fundamental statistical concerns discussed in the latter's documentation still apply.
