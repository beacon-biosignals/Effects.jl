using DataFrames
using Effects
using MixedModels
using StableRNGs
using Suppressor
using Test

rng = StableRNG(42)
x = rand(rng, 100)
data = (x=x, x2=1.5 .* x, y=rand(rng, [0, 1], 100), z=repeat('A':'T', 5))
model = @suppress fit(MixedModel, @formula(y ~ x + x2 + (1 | z)), data; progress=false)
dropped_idx = model.feterm.piv[end]
dropped_coef = coefnames(model)[dropped_idx]
kept_coef = last(fixefnames(model))

# due to the vagaries of numerical linear algebra and pivoting, it's
# not completely deterministic which of x and x2 gets dropped, though it
# will tend to be x2
design = Dict(Symbol(kept_coef) => [0])
eff = effects(design, model)
@test dropped_coef ∉ names(eff)
@test kept_coef ∈ names(eff)
@test !isnan(only(eff.err))
@test eff.y - eff.err ≈ eff.lower
@test eff.y + eff.err ≈ eff.upper

design = Dict(Symbol(dropped_coef) => [0])
eff = effects(design, model)
