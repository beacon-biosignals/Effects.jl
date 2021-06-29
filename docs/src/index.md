```@meta
CurrentModule = Effects
```

# Effects.jl

Regression models are useful but they can be tricky to interpret. Variable centering and contrast coding can obscure the meaning of main effects. Interaction terms, especially higher order ones, only increase the difficulty of interpretation. Here, we introduce Effects.jl which translates the fitted model, including estimated uncertainty, back into data space. Using Effects.jl, it is possible to generate effects plots that enable rapid visualization and interpretation of regression models.

The examples below demonstrate the use of Effects.jl with GLM.jl,
but they will work with any modeling package that is based on the [StatsModels.jl
formula](https://juliastats.org/StatsModels.jl/stable/formula/).
The second example is borrowed in no small part from [StandardizedPredictors.jl](https://beacon-biosignals.github.io/StandardizedPredictors.jl/dev/).

## The Effect of Contrast Coding

Let's consider a synthetic dataset of weights (in grams) for chicks feed different types of feed, with a single predictor `feed` (categorical, with three levels `A`, `B`, `C`). The simulated weights are based loosely on the R dataset `chickwts`.

```@example contrasts
using AlgebraOfGraphics, CairoMakie, DataFrames, Effects, GLM, StatsModels, Random
rng = MersenneTwister(42)
reps = 10
sd = 50
wtdat = DataFrame(feed = repeat(["A", "B", "C"], inner=reps),
                  weight=[180 .+ sd*randn(rng, reps);
                          220 .+ sd*randn(rng, reps);
                          300 .+ sd*randn(rng, reps)])
```

If we fit a linear model to this data using default treatment/dummy coding, then every term is significant:

```@example contrasts
mod_treat = lm(@formula(weight ~ 1 + feed), wtdat)
```

If on the other hand, we use effects (sum-to-zero) coding, then the term for feed `B` is no longer significant:

```@example contrasts
mod_eff = lm(@formula(weight ~ 1 + feed), wtdat; contrasts=Dict(:feed => EffectsCoding()))
```

This is in some sense unsurprising: the different coding schemes correspond to different hypotheses. In treatment coding, the hypothesis for the term `feed: B` is whether feed `B` differs from the reference level, feed `A`. In effects coding, the hypothesis is whether feed `B` differs from the mean across all levels. In more complicated models, the hypotheses being tested -- especially for interaction terms -- can become more complex and difficult to "read off" from the model summary.

In spite of these differences, these models make the same predictions about the data:

```@example contrasts
response(mod_treat) ≈ response(mod_eff)
```

At a deep level, these models are the actually same model, but with different parameterizations. In order to get a better view about what a model is saying about the data, abstracted away from the parameterization, we can see what the model looks like in data space. For that, we can use `Efffects.jl` to generate the *effects* that the model is capturing. We do this by specifying a (subset of the) design and creating a reference grid, then computing the model's prediction and associated error at those values.

The `effects` function will compute the reference grid for a fully-crossed design specified by a dictionary of values. As we only have one predictor in this dataset, the design is fully crossed.


```@example contrasts
design = Dict(:feed => unique(wtdat.feed))
eff_feed = effects(design, @formula(weight ~ 1 + feed), mod_eff;
                   contrasts=Dict(:feed => EffectsCoding()))
eff_feed
```

!!! warning
    You must specify the contrasts for the new formula such that they exactly match the contrasts used in fitting the original model. In the future, automatic extraction of the contrasts from the fitted model will be supported.

The effects table consists of four columns: the levels of the `feed` predictor specified in the design (`feed`), the prediction of the model at those levels (`weight`), the standard error of those predictions `err`, and the lower and upper edges of confidence interval of those predictions (`lower`, `upper`; computed using a normal approximation based on the standard error).

```@example contrasts
plt = data(eff_feed) * mapping(:feed, :weight) * (visual(Scatter) + mapping(:lower, :upper) * visual(Errorbars))
draw(plt)
```
## Interaction Terms in Effects

Let's consider a (slightly) synthetic dataset of weights for adolescents of
different ages, with predictors `age` (continuous, from 13 to 20) and `sex`, and
`weight` in pounds.  The weights are based loosely on the medians from the [CDC
growth charts](https://www.cdc.gov/growthcharts/html_charts/wtage.htm), which
show that the median male and female both start off around 100 pounds at age 13,
but by age 20 the median male weighs around 155 pounds while the median female
weighs around 125 pounds.

```@example centering
using AlgebraOfGraphics, CairoMakie, DataFrames, Effects, GLM, StatsModels, Random
rng = MersenneTwister(42)
growthdata = DataFrame(age=[13:20; 13:20],
                       sex=repeat(["male", "female"], inner=8),
                       weight=[range(100, 155; length=8); range(100, 125; length=8)] .+ randn(rng, 16))
```

In this dataset, there's obviously a main effect of sex: males are heavier than
females for every age except 13 years.  But if we run a basic linear regression, we
see something rather different:

```@example centering
mod_uncentered = lm(@formula(weight ~ 1 + sex * age), growthdata)
```

Is this model just a poor fit to the data? We can plot the effects and see that's not the case. For example purposes, we'll create a reference grid that does not correspond to a fully balanced design and call `effects!` to insert the effects-related columns. In particular, we'll take odd ages for males and even ages for females.

```@example centering
refgrid = copy(growthdata)
filter!(refgrid) do row
  return mod(row.age, 2) == (row.sex == "male")
end
effects!(refgrid, @formula(weight ~ 1 + sex * age), mod_uncentered)
```

Note that the column corresponding to the response variable `weight` has been overwritten with the effects prediction and that only the standard error is provided: the [`effects!`](@ref) method does less work than the [`effects`](@ref) convenience method.

We can add the confidence interval bounds in and plot our predictions:
```@example contrasts
refgrid[!, :lower] = @. refgrid.weight - 1.96 * refgrid.err
refgrid[!, :upper] = @. refgrid.weight + 1.96 * refgrid.err
sort!(refgrid, [:age])

plt = data(refgrid) * mapping(:age, :weight; lower=:lower, upper=:upper, color=:sex) *
      (visual(Lines) + visual(LinesFill))
draw(plt)
```

We can also add in the raw data to check the model fit:

```@example contrasts
draw(plt + data(growthdata) * mapping(:age, :weight; color=:sex) * visual(Scatter))
```

The model seems to be doing a good job. Indeed it is, and as pointed out in the [StandardizedPredictors.jl](https://beacon-biosignals.github.io/StandardizedPredictors.jl/dev/) docs, the problem is that we should center the `age` variable. While we're at it, we'll also set the contrasts for `sex` to be effects coded.

```@example centering
using StandardizedPredictors
contrasts = Dict(:age => Center(15), :sex => EffectsCoding())
mod_centered = lm(@formula(weight ~ 1 + sex * age), growthdata; contrasts=contrasts)
```

All of the estimates have now changed because the parameterization is completely different, but the predictions and thus the effects remain unchanged:

```@example centering
refgrid_centered = copy(growthdata)
effects!(refgrid_centered, @formula(weight ~ 1 + sex * age), mod_centered; contrasts=contrasts)
refgrid_centered[!, :lower] = @. refgrid_centered.weight - 1.96 * refgrid_centered.err
refgrid_centered[!, :upper] = @. refgrid_centered.weight + 1.96 * refgrid_centered.err
sort!(refgrid_centered, [:age])

plt = data(refgrid_centered) * mapping(:age, :weight; lower=:lower, upper=:upper, color=:sex) *
      (visual(Lines) + visual(LinesFill))
draw(plt)
```

Understanding lower-level terms in the presence of interactions can be particularly tricky, and effect plots are also useful for this. For example, if we want to examine the effect of `sex` at a *typical*  `age`, then we would need some way to reduce `age` to `typical` values. By default, `effects[!]` will take use the mean of all model terms not specified in the effects formula as representative values. Looking at `sex`, we see that

```@example centering
design = Dict(:sex => unique(growthdata.sex))
effects(design,  @formula(weight ~ 1 + sex), mod_uncentered)
```

correspond to the model's estimate of weights for each sex at the average age in the dataset. (Note that this is not quite the same as the average weight of each sex across all ages.) Like all effects predictions, this is invariant to contrast coding:

```@example centering
effects(design,  @formula(weight ~ 1 + sex), mod_centered; contrasts=contrasts)
```
