module EffectsGLMExt

using Effects

using GLM: AbstractGLM, Link, Link01, inverselink
using StatsModels: TableRegressionModel

# TODO: upstream a Link(::TableRegressionModel{<:AbstractGLM})
Effects._model_link(m::TableRegressionModel{<:AbstractGLM}, ::AutoInvLink) = Link(m.model)
Effects._model_link(m::AbstractGLM, ::AutoInvLink) = Link(m)
Effects._invlink_and_deriv(link::Link01, η) = inverselink(link, η)[1:2:3]  # (µ, 1 - µ, dμdη)
Effects._invlink_and_deriv(link::Link, η) = inverselink(link, η)[1:2]  # (µ, dμdη, NaN)

end # module
