module EffectsMixedModelsExt

using Effects
using MixedModels

using GLM: Link

Effects._link(m::GeneralizedLinearMixedModel) = Link(m.glm)

end # module
