module EffectsGLMExt

using Effects

using GLM: AbstractGLM, Link, mueta, linkinv
using StatsAPI: RegressionModel
using StatsModels: TableRegressionModel

# we keep RegressionModel so that this will also be used 
# for MixedModels.jl

Effects._link(m::TableRegressionModel{<:AbstractGLM}) = Link(m.model)

function Effects._difference_method!(eff::Vector{T}, err::Vector{T}, 
                                     model::RegressionModel, 
                                     ::AutoInvLink) where {T <: AbstractFloat}
    link = Effects._link(model)
    err .*= mueta.(link, eff)
    eff .= linkinv.(link, eff)

    return err
end


end # module
