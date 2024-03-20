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
    piv = view(model.feterm.piv, 1:model.feterm.rank)
    X = view(modelcols(form_typical, reference_grid), :, piv)
    eff = X * fixef(model)
    err = sqrt.(diag(X * vcov(model)[piv, piv] * X'))
    _difference_method!(eff, err, model, invlink)
    reference_grid[!, something(eff_col, _responsename(model))] = eff
    reference_grid[!, err_col] = err
    return reference_grid
end

end # module
