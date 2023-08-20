module EffectsGLMExt

using Effects

using GLM: AbstractGLM, Link, mueta, linkinv
using StatsAPI: RegressionModel
using StatsModels: TableRegressionModel

# TODO: upstream a Link(::TableRegressionModel{<:AbstractGLM})
_link(m::TableRegressionModel{<:AbstractGLM}) = Link(m.model)

function Effects._difference_method!(eff::Vector{T}, err::Vector{T},
                                     model::Union{TableRegressionModel{<:AbstractGLM},
                                                  AbstractGLM},
                                     ::AutoInvLink) where {T<:AbstractFloat}
    link = _link(model)
    err .*= mueta.(link, eff)
    eff .= linkinv.(link, eff)

    return err
end

end # module
