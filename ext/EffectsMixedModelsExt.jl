module EffectsMixedModelsExt

using DataFrames
using Effects
using MixedModels
using StatsBase

using Effects: typify, _difference_method!, _responsename
using LinearAlgebra: diag
using StatsModels: modelcols
using GLM: Link

Effects._model_link(m::GeneralizedLinearMixedModel, ::AutoInvLink) = Link(m)

function Effects.effects!(reference_grid::DataFrame, model::MixedModel;
                          eff_col=nothing, err_col=:err, typical=mean, invlink=identity,
                          vcov=StatsBase.vcov)
    # right now this is written for a RegressionModel and implicitly assumes
    # the existence of an appropriate formula method
    form = formula(model)
    form_typical = typify(reference_grid, form, modelmatrix(model); typical=typical)
    piv = view(model.feterm.piv, 1:(model.feterm.rank))
    X = view(modelcols(form_typical, reference_grid), :, piv)
    eff = X * fixef(model)
    err = sqrt.(diag(X * vcov(model)[piv, piv] * X'))
    _difference_method!(eff, err, model, invlink)
    reference_grid[!, something(eff_col, _responsename(model))] = eff
    reference_grid[!, err_col] = err
    return reference_grid
end

"""
    effects!(reference_grid::DataFrame, model::MixedModel, boot::MixedModelBootstrap;
             eff_col=nothing, err_col=:err, typical=mean, invlink=identity,
             lower_col=:lower, upper_col=:upper, level=nothing)

Use the results of a bootstrap to compute empirical error estimates
(instead of using the variance-covariance matrix).

!!! warning
    This method is experimental and may change in its defaults or
    disappear entirely in a future release!
"""
function Effects.effects!(reference_grid::DataFrame, model::MixedModel,
                          boot::MixedModelBootstrap;
                          eff_col=nothing, err_col=:err, typical=mean, invlink=identity,
                          lower_col=:lower, upper_col=:upper,
                          level=nothing)
    # right now this is written for a RegressionModel and implicitly assumes
    # the existence of an appropriate formula method
    form = formula(model)
    form_typical = typify(reference_grid, form, modelmatrix(model); typical=typical)
    # we don't need to worry about pivoting / fixef rank deficiency here
    # because the default -0.0 placeholder coefs still yield the correct predictions
    # it's the NaNs in the vcov that create problems, but this method sidesteps
    # the vcov computation, so we're all good!
    X = modelcols(form_typical, reference_grid)
    eff = X * coef(model)
    # each row is a bootstrap replicate
    # each column is a different row of X
    # this seems like a weird way to store things, until you remember
    # that we aggregate across replicates and thus want to take advantage
    # of column major storage
    boot_err = mapreduce(vcat, groupby(DataFrame(boot.β), :iter)) do gdf
        β = gdf[!, :β]
        return (X * β)'
    end

    err = map(std, eachcol(boot_err))
    _difference_method!(eff, err, model, invlink)
    reference_grid[!, something(eff_col, _responsename(model))] = eff
    reference_grid[!, err_col] = err

    # logic here is slightly different than for other methods
    # because we can compute empirical CIs instead of relying on a Wald
    # approximation
    if !isnothing(level)
        lower_tail = (1 - level) / 2
        upper_tail = 1 - lower_tail

        ci = map(eachcol(boot_err)) do col
            return quantile(col, [lower_tail, upper_tail])
        end

        reference_grid[!, lower_col] = first.(ci)
        reference_grid[!, upper_col] = last.(ci)
    end
    return reference_grid
end

"""
    effects(design::AbstractDict, model::MixedModel, boot::MixedModelBootstrap;
            eff_col=nothing, err_col=:err, typical=mean,
            lower_col=:lower, upper_col=:upper, invlink=identity,
            level=nothing)

Use the results of a bootstrap to compute empirical error estimates
(instead of using the variance-covariance matrix).

!!! warning
    This method is experimental and may change in its defaults or
    disappear entirely in a future release!
"""
function Effects.effects(design::AbstractDict, model::MixedModel, boot::MixedModelBootstrap;
                         eff_col=nothing, err_col=:err, typical=mean,
                         lower_col=:lower, upper_col=:upper, invlink=identity,
                         level=nothing)
    grid = expand_grid(design)
    dv = something(eff_col, _responsename(model))
    level = isnothing(level) ? 0.68 : level
    effects!(grid, model, boot; eff_col=dv, err_col, typical, invlink, level, lower_col,
             upper_col)
    return grid
end

end # module
