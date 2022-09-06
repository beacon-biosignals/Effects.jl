# Effects.jl
Effects Prediction for Linear and Generalized Linear models


[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://beacon-biosignals.github.io/Effects.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://beacon-biosignals.github.io/Effects.jl/dev)
[![Build Status][build-img]][build-url]
[![codecov](https://codecov.io/gh/beacon-biosignals/Effects.jl/branch/main/graph/badge.svg?token=AAK9265TXH)](https://codecov.io/gh/beacon-biosignals/Effects.jl)
[![DOI](https://zenodo.org/badge/323392527.svg)](https://zenodo.org/badge/latestdoi/323392527)

[build-img]: https://github.com/beacon-biosignals/Effects.jl/workflows/CI/badge.svg
[build-url]: https://github.com/beacon-biosignals/Effects.jl/actions

Regression is a foundational technique of statistical analysis, and many common statistical tests are based on regression models (e.g., ANOVA, t-test, correlation tests, etc.).
Despite the expressive power of regression models, users often prefer the simpler procedures because regression models themselves can be difficult to interpret.
Most notably, the interpretation of individual regression coefficients (including their magnitude, sign, and even significance) changes depending on the presence or even centering/contrast coding of other terms or interactions.
For instance, a common source of confusion in regression analysis is the meaning of the intercept coefficient. On its own, this coefficient corresponds to the grand mean of the independent variable, but in the presence of a contrast-coded categorical variable, it can correspond to the mean of the baseline level of that variable, the grand mean, or something else altogether, depending on the contrast coding scheme that is used.
Effects.jl provides a general-purpose tool for interpreting fitted regression models by projecting the effects of one or more terms in the model back into "data space", along with the associated uncertainty, fixing other the value of other terms at typical or user-specified values.
This makes it straightforward to interrogate the estimated effects of any predictor at any combination of other predictors' values. Because these effects are computed in data space, they can be plotted in parallel format to raw or aggregated data, enabling intuitive model interpretation and sanity checks.
