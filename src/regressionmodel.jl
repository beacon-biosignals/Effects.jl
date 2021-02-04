using Base.Iterators: product
using DataFrames
using LinearAlgebra
using Statistics
using StatsModels
using StatsBase
using Tables

"""
    _decompose_coefname(::String)

Split a coefficient name into  a set of constituent lower-order term names.

This is useful for comparing coefficient names constructed from formulae
with potentially different ordering of lower-order term names.
"""
_decompose_coefname(x) = Set(split(x, r"\s&\s"))

"""
    effects!(reference_grid::DataFrame, formula::FormulaTerm,
             model::RegressionModel;
             contrasts=Dict{Symbol,Any}(), err_col=:err, typical=mean)


Compute the `effects` as specified in `formula`.

Effects are the model predictions made using values given via the reference
grid for the terms specified in the formula. For terms present in the model,
but not in the formula and reference grid, then the typical value of those
predictors is used. (In other words, effects are conditional on the typical
value.) The function for computing typical values is specified via `typical`.
Note that this is also applied to categorical contrasts, thus yielding a
weighted average of the contrast.

The column corresponding to the response variable in the formula is overwritten
with the effects. Pointwise standard errors are written into the column
specified by `err_col`.

The reference grid must contain columns for all predictors in the formula.
(Interactions are computed automatically.) Contrasts must match the contrasts
used to fit the model; using other contrasts will lead to undefined behavior.

Note that including lower-level effects without interactions may lead to
misleading results because the higher-level interactions are replaced with
their typical values (i.e. the marginal typical value). In the future, this
may change to the product of the included lower-level effect with the typical
value of terms not in the model(i.e. the typical value conditioned on the
effects present).

The use of typical values for excluded effects differs from other approaches
such as "partial effects" used in R packages like [`remef`](https://github.com/hohenstein/remef/).
The R package [`effects`](https://cran.r-project.org/web/packages/effects/)
is similar in approach, but due to differing languages and licenses,
no source code was inspected and there is no attempt at API compatibility or
even similarity.

The approach for computing effect is based on the effects plots described here:

Fox, John (2003). Effect Displays in R for Generalised Linear Models.
Journal of Statistical Software. Vol. 8, No. 15
"""
function effects!(reference_grid::DataFrame, formula::FormulaTerm, model::RegressionModel;
                  contrasts=Dict{Symbol,Any}(), err_col=:err, typical=mean)
    # right now this is written for a RegressionModel and implicitly assumes
    # no link function
    sch = schema(formula, reference_grid, contrasts)
    form = apply_schema(formula, sch)
    depvar = form.lhs.sym
    y, X = modelcols(form, reference_grid)
    # note that we need columns, not terms now!
    refcols = _decompose_coefname.(coefnames(form.rhs))
    allcols = _decompose_coefname.(coefnames(model))
    nrows = size(X)[1]
    ncols = length(allcols)
    Xs = map(1:ncols) do idx
        if allcols[idx] in refcols
            return X[:, refcols .== Ref(allcols[idx])]
        else
            val = typical(modelmatrix(model)[:, idx])
            vec = Vector{Float64}(undef, nrows)
            vec .= val
            return vec
        end
    end
    X = hcat(Xs...)
    eff = X * coef(model)
    err = sqrt.(diag(X * vcov(model) * X'))
    # XXX DataFrames dependency
    reference_grid[!, depvar] = eff
    reference_grid[!, err_col] = err
    return reference_grid
end

"""
    effects(design::Union{NamedTuple, Dict},
            formula::FormulaTerm,
            model::RegressionModel;
            contrasts=Dict{Symbol,Any}(),
            err_col=:err, typical=mean,
            lower_col=:lower,
            upper_col=:upper)


Compute the `effects` as specified in `formula`.

This is a convenience wrapper for [`effects!`](@ref). Instead of specifying a
reference grid, a dictionary containing the levels/values of each predictor
is specified. This is then expanded into a reference grid representing a
fully-crossed design. Additionally, two extra columns are created representing
the lower and upper edge of the error band (i.e. [resp-err, resp+err]).
"""
function effects(design::NamedTuple, formula::FormulaTerm, model::RegressionModel;
                 contrasts=Dict{Symbol,Any}(), err_col=:err, typical=mean,
                 lower_col=:lower, upper_col=:upper)
    dv = formula.lhs.sym
    colnames =  [collect(keys(design)); dv]
    reference_grid = map(product(values(design)...)) do row
        rowdv = (row..., 0.0)
        return (; zip(colnames, rowdv)...)
    end
    reference_grid = effects!(DataFrame(reference_grid), formula, model; contrasts=contrasts,
                              err_col=err_col, typical=typical)
    # XXX DataFrames dependency
    reference_grid[!, lower_col] = reference_grid[!, dv] - reference_grid[!, err_col]
    reference_grid[!, upper_col] = reference_grid[!, dv] + reference_grid[!, err_col]
    return reference_grid
end

function effects(design::Dict, args...; kwargs...)
    nt = (;design...)
    return effects(nt, args...; kwargs...)
end
