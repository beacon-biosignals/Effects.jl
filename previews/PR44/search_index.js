var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"CurrentModule = Effects","category":"page"},{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [Effects]","category":"page"},{"location":"api/#Effects.effects!-Tuple{DataFrames.DataFrame, StatsAPI.RegressionModel}","page":"API","title":"Effects.effects!","text":"effects!(reference_grid::DataFrame, model::RegressionModel;\n         eff_col=nothing, err_col=:err, typical=mean, invlink=identity)\n\nCompute the effects as specified in formula.\n\nEffects are the model predictions made using values given via the reference grid. For terms present in the model, but not in the reference grid, then the typical value of those predictors is used. (In other words, effects are conditional on the typical value.) The function for computing typical values is specified via typical. Note that this is also applied to categorical contrasts, thus yielding an average of the contrast, weighted by the balance of levels in the data set used to fit the model.\n\nBy default, the column corresponding to the response variable in the formula is overwritten with the effects, but an alternative column for the effects can be specified by eff_col. Note that eff_col is determined first by trying StatsBase.responsename and then falling back to the string representation of the model's formula's left-hand side. For models with a transformed response, whether in the original formula specification or via hints/contrasts, the name will be the display name of the resulting term and not the original variable. This convention also has the advantage of highlighting another aspect of the underlying method: effects are computed on the scale of the transformed response. If this column does not exist, it is created. Pointwise standard errors are written into the column specified by err_col.\n\nnote: Note\nBy default (invlink=identity), effects are computed on the scale of the transformed response. For models with an explicit transformation, that transformation is the scale of the effects. For models with a link function, the scale of the effects is the link scale, i.e. after application of the link function. For example, effects for logitistic regression models are on the logit and not the probability scale.\n\nwarning: Warning\nIf the inverse link is specified via invlink, then effects and errors are computed on the original, untransformed scale via the delta method using automatic differentiation. This means that the invlink function must be differentiable and should not involve inplace operations.\n\nThe reference grid must contain columns for all predictors in the formula. (Interactions are computed automatically.) Contrasts must match the contrasts used to fit the model; using other contrasts will lead to undefined behavior.\n\nInteraction terms are computed in the same way for any regression model: as the product of the lower-order terms. Typical values of lower terms thus propagate up into the interaction term in the same way that any value would.\n\nThe use of typical values for excluded effects differs from other approaches such as \"partial effects\" used in R packages like remef. The R package effects is similar in approach, but due to differing languages and licenses, no source code was inspected and there is no attempt at API compatibility or even similarity.\n\nThe approach for computing effect is based on the effects plots described here:\n\nFox, John (2003). Effect Displays in R for Generalised Linear Models. Journal of Statistical Software. Vol. 8, No. 15\n\n\n\n\n\n","category":"method"},{"location":"api/#Effects.effects-Tuple{AbstractDict, StatsAPI.RegressionModel}","page":"API","title":"Effects.effects","text":"effects(design::AbstractDict, model::RegressionModel;\n        eff_col=nothing, err_col=:err, typical=mean,\n        lower_col=:lower, upper_col=:upper, invlink=identity)\n\nCompute the effects as specified by the design.\n\nThis is a convenience wrapper for effects!. Instead of specifying a reference grid, a dictionary containing the levels/values of each predictor is specified. This is then expanded into a reference grid representing a fully-crossed design. Additionally, two extra columns are created representing the lower and upper edge of the error band (i.e. [resp-err, resp+err]).\n\n\n\n\n\n","category":"method"},{"location":"api/#Effects.emmeans-Tuple{StatsAPI.RegressionModel}","page":"API","title":"Effects.emmeans","text":"emmeans(model::RegressionModel; eff_col=nothing, err_col=:err,\n        invlink=identity, levels=Dict(), dof=nothing)\n\nCompute estimated marginal means, a.k.a. least-square (LS) means for a model.\n\nBy default, emmeans are computed for each level of each categorical variable along with the means of continuous variables. For centered terms, the center is used instead of the mean. Alternative levels can be specified with levels.\n\ndof is used to compute the associated degrees of freedom for a particular margin. For regression models, the appropriate degrees of freedom are the same degrees of freedom as used for the Wald tests on the coefficients. These are typically the residual degrees of freedom (e.g., via dof_residual). If dof is a function, then it is called on model and filled elementwise into the dof column. Alternatively, dof can be specified as a single number or vector of appropriate length. For example, in a mixed effect model with a large number of observations, dof=Inf may be appropriate.\n\ninvlink, eff_col and err_col work exactly as in effects!.\n\nEstimated marginal means are closely related to effects and are also known as least-square means. The functionality here is a convenience wrapper for effects and maps onto the concept of least-square means as presented in e.g. the SAS documentation. There are several extensions available to estimated marginal means, related to when the marginalization occurs and how cells are weighted, but these are not currently supported. The documentation for the R package emmeans explains the background in more depth.\n\n\n\n\n\n","category":"method"},{"location":"api/#Effects.empairs-Tuple{StatsAPI.RegressionModel}","page":"API","title":"Effects.empairs","text":"empairs(model::RegressionModel; eff_col=nothing, err_col=:err,\n        invlink=identity, levels=Dict(), dof=nothing, padjust=identity)\nempairs(df::AbstractDataFrame; eff_col, err_col=:err, padjust=identity)\n\nCompute pairwise differences of estimated marginal means.\n\nThe method for AbstractDataFrame acts on the results of emmeans, while the method for RegressionModel is a convenience wrapper that calls emmeans internally.\n\nThe keyword arguments are generally the same as emmeans.\n\nBy default, pairs are computed for all combinations of categorical variables and the means/centers of continuous variables. The contrast for a pair \"a\" vs \"b\" is represented in the column for the contrasted variable with a > b. For variables that are the same within a pair (e.g., a continuous variable), the sole value is displayed as is.\n\nIf dof is not nothing, then p-values are also computed using a t-distribution with the resulting degrees of freedom. The results for dof=Inf correspond to using a z-distribution, but the column names in the returned dataframe remain t and Pr(>|t|).\n\nIf padjust is provided, then it is used to compute adjust the p-values for multiple comparisons. MultipleTesting.jl provides a number of useful possibilities for this.\n\nnote: Note\npadjust is silently ignored if dof is not provided.\n\nwarning: Warning\nThis feature is experimental and the precise column names and presentation of contrasts/differences may change without being considered breaking.\n\nwarning: Warning\nThe use of invlink is subject to a number of interpretation subtleties. The EM means are computed on the scale of the linear predictor, then transformed to the scale of invlink. The associated errors on the transformed scale are computed via the difference method. These estimates and errors are then used to compute the pairs. Test statistics are computed on the scale of these pairs. In general, these will not be the same as test statistics computed on the original scale. These subleties are discussed in the documentation for the R package emmeans.\n\n\n\n\n\n","category":"method"},{"location":"api/#Effects.expand_grid-Tuple{Any}","page":"API","title":"Effects.expand_grid","text":"expand_grid(design)\n\nCompute a fully crossed reference grid.\n\ndesign can be either a NamedTuple or a Dict, where the keys represent the variables, i.e., column names in the resulting grid and the values are vectors of possible values that each variable can take on.\n\njulia> expand_grid(Dict(:x => [\"a\", \"b\"], :y => 1:3))\n6×2 DataFrame\n Row │ y      x\n     │ Int64  String\n─────┼───────────────\n   1 │     1  a\n   2 │     2  a\n   3 │     3  a\n   4 │     1  b\n   5 │     2  b\n   6 │     3  b\n\n\n\n\n\n","category":"method"},{"location":"api/#Effects.pooled_sem-Tuple","page":"API","title":"Effects.pooled_sem","text":"pooled_sem(sems...)\n\nCompute the pooled standard error of the mean.\n\nThis corresponds to the square root of the sum of the the squared SEMs, which in turn corresponds to the weighted root-mean-square of the underlying standard deviations.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Effects","category":"page"},{"location":"#Effects.jl","page":"Home","title":"Effects.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Regression models are useful but they can be tricky to interpret. Variable centering and contrast coding can obscure the meaning of main effects. Interaction terms, especially higher order ones, only increase the difficulty of interpretation. Here, we introduce Effects.jl which translates the fitted model, including estimated uncertainty, back into data space. Using Effects.jl, it is possible to generate effects plots that enable rapid visualization and interpretation of regression models.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The examples below demonstrate the use of Effects.jl with GLM.jl, but they will work with any modeling package that is based on the StatsModels.jl formula. The second example is borrowed in no small part from StandardizedPredictors.jl.","category":"page"},{"location":"#The-Effect-of-Contrast-Coding","page":"Home","title":"The Effect of Contrast Coding","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Let's consider a synthetic dataset of weights (in grams) for chicks feed different types of feed, with a single predictor feed (categorical, with three levels A, B, C). The simulated weights are based loosely on the R dataset chickwts.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using AlgebraOfGraphics, CairoMakie, DataFrames, Effects, GLM, StatsModels, StableRNGs\nrng = StableRNG(42)\nreps = 5\nsd = 50\nwtdat = DataFrame(; feed=repeat([\"A\", \"B\", \"C\"], inner=reps),\n                  weight=[180 .+ sd * randn(rng, reps);\n                          220 .+ sd * randn(rng, reps);\n                          300 .+ sd * randn(rng, reps)])","category":"page"},{"location":"","page":"Home","title":"Home","text":"If we fit a linear model to this data using default treatment/dummy coding, then the coefficient corresponding of feed C is significant.","category":"page"},{"location":"","page":"Home","title":"Home","text":"mod_treat = lm(@formula(weight ~ 1 + feed), wtdat)","category":"page"},{"location":"","page":"Home","title":"Home","text":"If on the other hand, we use effects (sum-to-zero) coding, then the coefficient for feed C is no longer significant:","category":"page"},{"location":"","page":"Home","title":"Home","text":"mod_eff = lm(@formula(weight ~ 1 + feed), wtdat; contrasts=Dict(:feed => EffectsCoding()))","category":"page"},{"location":"","page":"Home","title":"Home","text":"This is in some sense unsurprising: the different coding schemes correspond to different hypotheses. In treatment coding, the hypothesis for the coefficient feed: C is whether feed C differs from the reference level, feed A. In effects coding, the hypothesis is whether feed C differs from the mean across all levels. In more complicated models, the hypotheses being tested – especially for interaction terms – can become more complex and difficult to \"read off\" from the model summary.","category":"page"},{"location":"","page":"Home","title":"Home","text":"In spite of these differences, these models make the same predictions about the data:","category":"page"},{"location":"","page":"Home","title":"Home","text":"response(mod_treat) ≈ response(mod_eff)","category":"page"},{"location":"","page":"Home","title":"Home","text":"At a deep level, these models are the actually same model, but with different parameterizations. In order to get a better view about what a model is saying about the data, abstracted away from the parameterization, we can see what the model looks like in data space. For that, we can use Effects.jl to generate the effects that the model is capturing. We do this by specifying a (subset of the) design and creating a reference grid, then computing the model's prediction and associated error at those values.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The effects function will compute the reference grid for a crossed design specified by a dictionary of values. As we only have one predictor in this dataset, the design is completely crossed.","category":"page"},{"location":"","page":"Home","title":"Home","text":"design = Dict(:feed => unique(wtdat.feed))\neff_feed = effects(design, mod_eff)","category":"page"},{"location":"","page":"Home","title":"Home","text":"The effects table consists of four columns: the levels of the feed predictor specified in the design (feed), the prediction of the model at those levels (weight), the standard error of those predictions err, and the lower and upper edges of confidence interval of those predictions (lower, upper; computed using a normal approximation based on the standard error).","category":"page"},{"location":"","page":"Home","title":"Home","text":"Because the effects are on the data (response) scale, they are invariant to parameterization:","category":"page"},{"location":"","page":"Home","title":"Home","text":"effects(design, mod_treat)","category":"page"},{"location":"","page":"Home","title":"Home","text":"We can also compute effects on a subset of the levels in the data:","category":"page"},{"location":"","page":"Home","title":"Home","text":"effects(Dict(:feed => [\"A\"]), mod_eff)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note that in this case, we computed the average effect of the level A, which is not directly encoded in any single model coefficient in this parameterization / contrast coding.","category":"page"},{"location":"","page":"Home","title":"Home","text":"We can of course use the estimated effects to get a nice visual of the data:","category":"page"},{"location":"","page":"Home","title":"Home","text":"plt = data(eff_feed) * mapping(:feed, :weight) * (visual(Scatter) + mapping(:err) * visual(Errorbars))\ndraw(plt)","category":"page"},{"location":"#Interaction-Terms-in-Effects","page":"Home","title":"Interaction Terms in Effects","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Let's consider a (slightly) synthetic dataset of weights for adolescents of different ages, with predictors age (continuous, from 13 to 20) and sex, and weight in pounds.  The weights are based loosely on the medians from the CDC growth charts, which show that the median male and female both start off around 100 pounds at age 13, but by age 20 the median male weighs around 155 pounds while the median female weighs around 125 pounds.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using AlgebraOfGraphics, CairoMakie, DataFrames, Effects, GLM, StatsModels, StableRNGs\nrng = StableRNG(42)\ngrowthdata = DataFrame(; age=[13:20; 13:20],\n                       sex=repeat([\"male\", \"female\"], inner=8),\n                       weight=[range(100, 155; length=8); range(100, 125; length=8)] .+ randn(rng, 16))","category":"page"},{"location":"","page":"Home","title":"Home","text":"In this dataset, there's obviously a main effect of sex: males are heavier than females for every age except 13 years.  But if we run a basic linear regression, we see something rather different:","category":"page"},{"location":"","page":"Home","title":"Home","text":"mod_uncentered = lm(@formula(weight ~ 1 + sex * age), growthdata)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Is this model just a poor fit to the data? We can plot the effects and see that's not the case. For example purposes, we'll create a reference grid that does not correspond to a fully balanced design and call effects! to insert the effects-related columns. In particular, we'll take odd ages for males and even ages for females.","category":"page"},{"location":"","page":"Home","title":"Home","text":"refgrid = copy(growthdata)\nfilter!(refgrid) do row\n    return mod(row.age, 2) == (row.sex == \"male\")\nend\neffects!(refgrid, mod_uncentered)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note that the column corresponding to the response variable weight has been overwritten with the effects prediction and that only the standard error is provided: the effects! method does less work than the effects convenience method.","category":"page"},{"location":"","page":"Home","title":"Home","text":"We can add the confidence interval bounds in and plot our predictions:","category":"page"},{"location":"","page":"Home","title":"Home","text":"refgrid[!, :lower] = @. refgrid.weight - 1.96 * refgrid.err\nrefgrid[!, :upper] = @. refgrid.weight + 1.96 * refgrid.err\nsort!(refgrid, [:age])\n\nplt = data(refgrid) * mapping(:age, :weight; lower=:lower, upper=:upper, color=:sex) *\n      (visual(Lines) + visual(LinesFill))\ndraw(plt)","category":"page"},{"location":"","page":"Home","title":"Home","text":"We can also add in the raw data to check the model fit:","category":"page"},{"location":"","page":"Home","title":"Home","text":"draw(plt + data(growthdata) * mapping(:age, :weight; color=:sex) * visual(Scatter))","category":"page"},{"location":"","page":"Home","title":"Home","text":"The model seems to be doing a good job. Indeed it is, and as pointed out in the StandardizedPredictors.jl docs, the problem is that we should center the age variable. While we're at it, we'll also set the contrasts for sex to be effects coded.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using StandardizedPredictors\ncontrasts = Dict(:age => Center(15), :sex => EffectsCoding())\nmod_centered = lm(@formula(weight ~ 1 + sex * age), growthdata; contrasts=contrasts)","category":"page"},{"location":"","page":"Home","title":"Home","text":"All of the estimates have now changed because the parameterization is completely different, but the predictions and thus the effects remain unchanged:","category":"page"},{"location":"","page":"Home","title":"Home","text":"refgrid_centered = copy(growthdata)\neffects!(refgrid_centered, mod_centered)\nrefgrid_centered[!, :lower] = @. refgrid_centered.weight - 1.96 * refgrid_centered.err\nrefgrid_centered[!, :upper] = @. refgrid_centered.weight + 1.96 * refgrid_centered.err\nsort!(refgrid_centered, [:age])\n\nplt = data(refgrid_centered) * mapping(:age, :weight; lower=:lower, upper=:upper, color=:sex) *\n      (visual(Lines) + visual(LinesFill))\ndraw(plt)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Understanding lower-level terms in the presence of interactions can be particularly tricky, and effect plots are also useful for this. For example, if we want to examine the effect of sex at a typical age, then we would need some way to reduce age to typical values. By default, effects[!] will take use the mean of all model terms not specified in the effects formula as representative values. Looking at sex, we see that","category":"page"},{"location":"","page":"Home","title":"Home","text":"design = Dict(:sex => unique(growthdata.sex))\neffects(design, mod_uncentered)","category":"page"},{"location":"","page":"Home","title":"Home","text":"correspond to the model's estimate of weights for each sex at the average age in the dataset. Like all effects predictions, this is invariant to contrast coding:","category":"page"},{"location":"","page":"Home","title":"Home","text":"effects(design, mod_centered)","category":"page"},{"location":"#Categorical-Variables-and-Typical-Values","page":"Home","title":"Categorical Variables and Typical Values","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"We can also compute effects for the different ages at typical values of sex for this dataset. For categorical variables, the typical values are computed on the contrasts themselves. For balanced datasets and the default use of the mean as typical values, this reflects the average across all levels of the categorical variable.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For our model using the centered parameterization and effects coding, the effect at the center (age = 15) at typical values of sex is simply the intercept: the mean weight across both sexes at age 15.","category":"page"},{"location":"","page":"Home","title":"Home","text":"effects(Dict(:age => [15]), mod_centered)","category":"page"},{"location":"","page":"Home","title":"Home","text":"If we had an imbalanced dataset, then averaging across all values of the contrasts would be weighted towards the levels (here: sex) with more observations. This makes sense: the category with more observations is in some sense more \"typical\" for that dataset.","category":"page"},{"location":"","page":"Home","title":"Home","text":"# create imbalance\ngrowthdata2 = growthdata[5:end, :]\nmod_imbalance = lm(@formula(weight ~ 1 + sex * age), growthdata2; contrasts=contrasts)\neffects(Dict(:age => [15]), mod_imbalance)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Finally, we note that the user can specify alternative functions for computing the typical values. For example, specifying the mode for the imbalanced dataset results in effects estimates for the most frequent level of sex, i.e. female:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using StatsBase\neffects(Dict(:age => [15]), mod_imbalance; typical=mode)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note that these should be scalar valued functions, so we can use minimum or maximum but not extrema:","category":"page"},{"location":"","page":"Home","title":"Home","text":"effects(Dict(:sex => [\"female\", \"male\"]), mod_imbalance; typical=maximum)","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Test\n@test_throws ArgumentError effects(Dict(:sex => [\"female\", \"male\"]), mod_imbalance; typical=extrema)","category":"page"},{"location":"emmeans/#Estimated-Marginal-a.k.a.-Least-Square-Means","page":"Estimated Marginal Means","title":"Estimated Marginal a.k.a. Least Square Means","text":"","category":"section"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"At their simplest, estimated marginal means, a.k.a least-square means, are just effects: predictions and associated errors computed from a model at specific points on a reference grid. The \"marginal\" here refers to estimation at the margins of the table, i.e. either averaging over or computing at specific values of other variables, often the means of other variables. Effects.jl provides a convenient interface for computing EM means at all levels of categorical variables and the means of any continuous covariates. An example will make this clear:","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"using DataFrames, Effects, GLM, StatsModels, StableRNGs, StandardizedPredictors\nrng = StableRNG(42)\ngrowthdata = DataFrame(; age=[13:20; 13:20],\n                       sex=repeat([\"male\", \"female\"], inner=8),\n                       weight=[range(100, 155; length=8); range(100, 125; length=8)] .+ randn(rng, 16))\nmodel_scaled = lm(@formula(weight ~ 1 + sex * age), growthdata;\n                  contrasts=Dict(:age => ZScore(), :sex => DummyCoding()))\n\nemmeans(model_scaled)","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"(This is the same example data as Interaction Terms in Effects.)","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"Notice that we could have achieved the same result by using effects and specifying the levels manually:","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"design = Dict(:sex => [\"female\", \"male\"], :age => 16.5)\ngrid = expand_grid(design)\neffects!(grid, model_scaled)","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"emmeans is primarily a convenience function for computing effects on a convenient, pre-defined reference grid. Notably, emmeans includes information by default about all variables, even the ones analyzed only at their typical values. This differs from effects!:","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"design = Dict(:sex => [\"female\", \"male\"])\ngrid = expand_grid(design)\neffects!(grid, model_scaled)","category":"page"},{"location":"emmeans/#Pairwise-Comparisons","page":"Estimated Marginal Means","title":"Pairwise Comparisons","text":"","category":"section"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"EM means are also useful for post-hoc examination of pairwise differences. This is implemented via the empairs function:","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"empairs(model_scaled)","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"Or for a more advanced case:","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"using MixedModels\nkb07 = MixedModels.dataset(:kb07)\nmixed_form = @formula(rt_trunc ~ 1 + spkr * prec * load + (1|item) + (1|subj))\nmixed_model = fit(MixedModel, mixed_form, kb07; progress=false)\nempairs(mixed_model)","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"If we provide a method for computing the degrees of freedom or an appropriate value, then empairs will also generate test statistics:","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"empairs(model_scaled; dof=dof_residual)","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"These degrees of freedom correspond to the denominator degrees of freedom in the associated F-tests.","category":"page"},{"location":"emmeans/#Mixed-effects-Models","page":"Estimated Marginal Means","title":"Mixed-effects Models","text":"","category":"section"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"For mixed-effects models, the denominator degrees of freedom are not clearly defined[GLMMFAQ] (nor is it clear that the asymptotic distribution is even F for anything beyond a few special cases[Bates2006]). The fundamental problem is that mixed models have in some sense \"more\" degrees of freedom than you would expect from naively counting parameters – the conditional modes (BLUPs) aren't technically parameters, but do somehow contribute to increasing the amount of wiggle-room that the model has for adapting to the data. We can try a few different approaches and see how they differ.","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"We'll use a simplified version of the model above:","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"mixed_form2 = @formula(rt_trunc ~ 1 + prec * load + (1|item) + (1|subj))\nmixed_model2 = fit(MixedModel, mixed_form2, kb07; progress=false)","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"the default definition in MixedModels.jl for dof_residual(model) is simply nobs(model) - dof(model)","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"empairs(mixed_model2; dof=dof_residual)","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"the leverage formula, given by nobs(model) - sum(leverage(model)) (for classical linear models, this is exactly the same as (1))","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"empairs(mixed_model2; dof=nobs(mixed_model) - sum(leverage(mixed_model)))","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"counting each conditional mode as a parameter (this is the most conservative value because it has the smallest degrees of freedom)","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"empairs(mixed_model2; dof=nobs(mixed_model) - sum(size(mixed_model)[2:3]))","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"the infinite degrees of freedom (i.e. interpret t as z and F as chi^2 )","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"empairs(mixed_model2; dof=Inf)","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"These values are all very similar to each other, because the t distribution rapidly converges to the z distribution for t  30 and so amount of probability mass in the tails does not change much for a model with more than a thousand observations and 30+ levels of each grouping variable.","category":"page"},{"location":"emmeans/#Multiple-Comparisons-Correction","page":"Estimated Marginal Means","title":"Multiple Comparisons Correction","text":"","category":"section"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"A perhaps more important concern is inflation of Type-1 error rate through multiple testing. We can also provide a correction method to be applied to the p-values.","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"using MultipleTesting\nbonferroni(pvals) = adjust(PValues(pvals), Bonferroni())\nempairs(mixed_model2; dof=Inf, padjust=bonferroni)","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"[GLMMFAQ]: Why doesn’t lme4 display denominator degrees of freedom/p values? What other options do I have?","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"[Bates2006]: lmer, p-values and all that","category":"page"},{"location":"emmeans/#A-Final-Note","page":"Estimated Marginal Means","title":"A Final Note","text":"","category":"section"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"There are many subleties in more advanced applications of EM means. These are discussed at length in the documentation for the R package emmeans. For example:","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"the impact of transformations and link functions\nderived covariates, e.g., including both x and x^2 in a model\nweighting of cells in computing means","category":"page"},{"location":"emmeans/","page":"Estimated Marginal Means","title":"Estimated Marginal Means","text":"All of these problems are fundamental to EM means as a technique and not particular to the software implementation. Effects.jl does not yet support everything that R's emmeans does, but all of the fundamental statistical concerns discussed in the latter's documentation still apply.","category":"page"}]
}
