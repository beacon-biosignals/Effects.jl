# determine AutoInvLink automatically if a package with an appropriate extension
# is loaded

"""
    AutoInvLink

Singleton type indicating that the inverse link should be automatically
determined from the model type.

!!! compat "Julia 1.9"
    Automatic inverse link determination is implemented using package
    extensions, which are available beginning in Julia 1.9.
    
An error is thrown if the inverse link cannot be determined. This will
always occur with Julia versions prior to 1.9, and will otherwise occur
when no extension has been loaded that specifies the link function for
the model type.

Currently, this is only implemented for GLM.jl and MixedModels.jl
"""
struct AutoInvLink end

"""
    effects!(reference_grid::DataFrame, model::RegressionModel;
             eff_col=nothing, err_col=:err, typical=mean, invlink=identity,
             vcov=StatsBase.vcov)

Compute the `effects` as specified in `formula`.

Effects are the model predictions made using values given via the reference
grid. For terms present in the model, but not in the reference grid, then
the typical value of those predictors is used. (In other words, effects are
conditional on the typical value.) The function for computing typical values
is specified via `typical`. Note that this is also applied to categorical contrasts,
thus yielding an average of the contrast, weighted by the balance of levels in the data
set used to fit the model.

`typical` can either be a single, scalar-valued function (e.g. `mean`) or a dictionary
matching term symbols to scalar-valued functions. The use of a dictionary
allows specifying different `typical` functions for different input variables.
In this case, `typical` functions must be provided for all term variables
except the intercept. If there is a single term that should be "typified"
differently than others, then the use of `DataStructures.DefaultDict` may
be useful to create a default `typical` value with only the exception
explicitly specified. For example:

```julia
using DataStructures
typical = DefaultDict(() -> mean)  # default to x -> mean(x)
typical[:sex] = v -> 0.0           # typical value for :sex
```

By default, the column corresponding to the response variable in the formula
is overwritten with the effects, but an alternative column for the effects can
be specified by `eff_col`. Note that `eff_col` is determined first by trying
`StatsBase.responsename` and then falling back to the `string` representation
of the model's formula's left-hand side. For models with a transformed response,
whether in the original formula specification or via hints/contrasts, the name
will be the display name of the resulting term and not the original variable.
This convention also has the advantage of highlighting another aspect of the
underlying method: effects are computed on the scale of the transformed response.
If this column does not exist, it is created.
Pointwise standard errors are written into the column specified by `err_col`.

!!! note
    By default (`invlink=identity`), effects are computed on the scale of the
    transformed response. For models with an explicit transformation, that
    transformation is the scale of the effects. For models with a link function,
    the scale of the effects is the _link_ scale, i.e. after application of the
    link function. For example, effects for logitistic regression models are on
    the logit and not the probability scale.

!!! warning
    If the inverse link is specified via `invlink`, then effects and errors are
    computed on the original, untransformed scale via the delta method using
    automatic differentiation. This means that the `invlink` function must be
    differentiable and should not involve inplace operations.

On Julia versions 1.9 or later, the special singleton value `AutoInvLink()`
can be used to specify that the appropriate inverse link should be determined
automatically. In that case, a direct or analytic computation of the derivative
is used when possible.

Effects are computed using the model's variance-covariance matrix, which is
computed by default using `StatsBas.vcov`. Alternative methods such as the
sandwich estimator or robust estimators can be used by specifying `vcov`,
which should be a function of a single argument (the model) returning
the estimated variance-covariance matrix.
[Vcov.jl](https://github.com/FixedEffects/Vcov.jl) and [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl)
provide several possibilities as functions of multiple arguments and so it
is necessary to curry when using these functions. For example
```julia
using Vcov
myvcov(x) = Base.Fix2(vcov, Vcov.robust())
```

The reference grid must contain columns for all predictors in the formula.
(Interactions are computed automatically.) Contrasts must match the contrasts
used to fit the model; using other contrasts will lead to undefined behavior.

Interaction terms are computed in the same way for any regression model: as the
product of the lower-order terms. Typical values of lower terms thus propagate up
into the interaction term in the same way that any value would.

The use of typical values for excluded effects differs from other approaches
such as "partial effects" used in R packages like [`remef`](https://github.com/hohenstein/remef/).
The R package [`effects`](https://cran.r-project.org/web/packages/effects/)
is similar in approach, but due to differing languages and licenses,
no source code was inspected and there is no attempt at API compatibility or
even similarity.

The approach for computing effect is based on the effects plots described here:

Fox, John (2003). Effect Displays in R for Generalised Linear Models.
Journal of Statistical Software. Vol. 8, No. 15

See also [`AutoInvLink`](@ref).
"""
function effects!(reference_grid::DataFrame, model::RegressionModel;
                  eff_col=nothing, err_col=:err, typical=mean, invlink=identity,
                  vcov=StatsBase.vcov)
    # right now this is written for a RegressionModel and implicitly assumes
    # the existence of an appropriate formula method
    form = formula(model)
    form_typical = typify(reference_grid, form, modelmatrix(model); typical=typical)
    X = modelcols(form_typical, reference_grid)
    eff = X * coef(model)
    err = sqrt.(diag(X * vcov(model) * X'))
    _difference_method!(eff, err, model, invlink)
    reference_grid[!, something(eff_col, _responsename(model))] = eff
    reference_grid[!, err_col] = err
    return reference_grid
    # XXX remove DataFrames dependency
    # this doesn't work for a DataFrame and isn't mutating
    # return (; reference_grid..., depvar => eff, err_col => err)
end

# TODO: support the transformation method
# in addition to the difference method
# xref https://github.com/JuliaStats/GLM.jl/blob/c13577eaf3f418c58020534dd407532ee57f219b/src/glmfit.jl#L773-L783

_invlink_and_deriv(invlink, η) = (invlink(η), ForwardDiff.derivative(invlink, η))
_invlink_and_deriv(::typeof(identity), η) = (η, 1)
# this isn't the best name because it sometimes returns the inverse link and sometimes the link (Link())
# for now, this is private API, but we should see how this goes and whether we can make it public API
# so local extensions (instead of Package-Extensions) are better supported 
_model_link(::RegressionModel, invlink::Function) = invlink
function _model_link(::RegressionModel, ::AutoInvLink)
    @static if VERSION < v"1.9"
        @error "AutoInvLink requires extensions and is thus not available on Julia < 1.9."
    end
    throw(ArgumentError("No appropriate extension is loaded for automatic " *
                        "determination of the inverse link for this model type"))
end

function _difference_method!(eff::Vector{T}, err::Vector{T},
                             m::RegressionModel,
                             invlink) where {T<:AbstractFloat}
    link = _model_link(m, invlink)
    @inbounds for i in eachindex(eff, err)
        μ, dμdη = _invlink_and_deriv(link, eff[i])
        err[i] *= dμdη
        eff[i] = μ
    end
    return eff, err
end

"""
    expand_grid(design)

Compute a fully crossed reference grid.

`design` can be either a `NamedTuple` or a `Dict`, where the keys represent
the variables, i.e., column names in the resulting grid and the values are
vectors of possible values that each variable can take on.

```julia
julia> expand_grid(Dict(:x => ["a", "b"], :y => 1:3))
6×2 DataFrame
 Row │ y      x
     │ Int64  String
─────┼───────────────
   1 │     1  a
   2 │     2  a
   3 │     3  a
   4 │     1  b
   5 │     2  b
   6 │     3  b
```
"""
function expand_grid(design)
    colnames = tuple(keys(design)...)
    rowtab = NamedTuple{colnames}.(product(values(design)...))

    return DataFrame(vec(rowtab))
end

"""
    effects(design::AbstractDict, model::RegressionModel;
            eff_col=nothing, err_col=:err, typical=mean,
            lower_col=:lower, upper_col=:upper, invlink=identity,
            vcov=StatsBase.vcov, level=nothing)

Compute the `effects` as specified by the `design`.

This is a convenience wrapper for [`effects!`](@ref). Instead of specifying a
reference grid, a dictionary containing the levels/values of each predictor
is specified. This is then expanded into a reference grid representing a
fully-crossed design. Additionally, two extra columns are created representing
the lower and upper edges of the confidence interval.
The default `level=nothing` corresponds to [resp-err, resp+err], while `level=0.95`
corresponds to the 95% confidence interval.

See also [`AutoInvLink`](@ref).
"""
function effects(design::AbstractDict, model::RegressionModel;
                 eff_col=nothing, err_col=:err, typical=mean,
                 lower_col=:lower, upper_col=:upper, invlink=identity,
                 vcov=StatsBase.vcov, level=nothing)
    grid = expand_grid(design)
    dv = something(eff_col, _responsename(model))
    effects!(grid, model; eff_col=dv, err_col, typical, invlink, vcov)
    # level=0.68 is approximately one standard error, but it's just enough
    # off to create all sorts of problems with tests and
    # cause all our tests to fail, which means it would create problems for
    # users, so we special case it to maintain legacy behavior
    level_scale = isnothing(level) ? 1 : sqrt(quantile(Chisq(1), level))
    # XXX DataFrames dependency
    grid[!, lower_col] = grid[!, dv] - grid[!, err_col] * level_scale
    grid[!, upper_col] = grid[!, dv] + grid[!, err_col] * level_scale
    return grid
    # up_low = let dv = getproperty(reference_grid, dv), err = getproperty(reference_grid, err_col)
    #     (; lower_col => dv .- err, upper_col => dv .+ err)
    # end
    # return (; reference_grid..., up_low...)
end

function _responsename(model::RegressionModel)
    return try
        responsename(model)
    catch ex
        # why not specialize on MethodError here?
        # well StatsBase defines stubs for all functions in its API
        # that just use `error()`
        _responsename(formula(model))
    end
end

function _responsename(f::FormulaTerm)
    return string(f.lhs)
end
