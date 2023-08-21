module EffectsMixedModelsExt

using Effects
using MixedModels
using GLM: Link

Effects._model_link(m::GeneralizedLinearMixedModel, ::AutoInvLink) = Link(m)

end # module
