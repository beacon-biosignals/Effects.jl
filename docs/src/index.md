```@meta
CurrentModule = Effects
```

# Effects.jl

# StandardizedPredictors

Regression models are useful but they can be tricky to interpret. Variable centering and contrast coding can obscure the meaning of main effects. Interaction terms, especially higher order ones, only increase the difficulty of interpretation. Here, we introduce Effects.jl which translates the fitted model, including estimated uncertainty, back into data space. Using Effects.jl, it is possible to generate effects plots that enable rapid visualization and interpretation of regression models.

The examples below demonstrate the use of Effects.jl with GLM.jl,
but they will work with any modeling package that is based on the [StatsModels.jl
formula](https://juliastats.org/StatsModels.jl/stable/formula/).
These examples are borrowed in no small part from [StandardizedPredictors.jl](https://beacon-biosignals.github.io/StandardizedPredictors.jl/dev/).
