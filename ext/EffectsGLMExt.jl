module EffectsGLMExt

using Effects

using GLM: AbstractGLM, Link, Link01, linkinv, mueta
using Effects.StatsModels: TableRegressionModel

# TODO: upstream a Link(::TableRegressionModel{<:AbstractGLM})
Effects._model_link(m::TableRegressionModel{<:AbstractGLM}, ::AutoInvLink) = Link(m.model)
Effects._model_link(m::AbstractGLM, ::AutoInvLink) = Link(m)
Effects._invlink_and_deriv(link::Link01, η) = (linkinv(link, η), mueta(link, η))
Effects._invlink_and_deriv(link::Link, η) = (linkinv(link, η), mueta(link, η))

end # module
