module EffectsMixedModelsExt

using Effects
using MixedModels

function Effects._difference_method!(eff::Vector{T}, err::Vector{T}, 
                                     model::GeneralizedLinearMixedModel, 
                                     ::AutoInvLink) where {T <: AbstractFloat}
    link = Link(model)
    err .*= mueta.(link, eff)
    eff .= linkinv.(link, eff)
    return err
end

end # module
