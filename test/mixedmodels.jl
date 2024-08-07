using DataFrames
using Effects
using MixedModels
using StableRNGs
using Suppressor
using Test

rng = StableRNG(42)
x = rand(rng, 100)
a = rand(rng, 100)
data = (; x, x2=1.5 .* x, a, y=rand(rng, [0, 1], 100), g=repeat('A':'T', 5))
model = @suppress fit(MixedModel, @formula(y ~ a + x + x2 + (1 | g)), data; progress=false)
dropped_idx = model.feterm.piv[end]
dropped_coef = coefnames(model)[dropped_idx]
kept_coef = last(fixefnames(model))

# due to the vagaries of numerical linear algebra and pivoting, it's
# not completely deterministic which of x and x2 gets dropped, though it
# will tend to be x2
design = Dict(Symbol(kept_coef) => [1], :a => [2])
eff = effects(design, model)
@test dropped_coef ∉ names(eff)
@test kept_coef ∈ names(eff)
@test all(!isnan, eff.err)
@test eff.y - eff.err ≈ eff.lower
@test eff.y + eff.err ≈ eff.upper

β = fixef(model)
# note that in the design above, we had a = 2
# β[1] = intercept
# β[2] = a, multiply by 2 because that's what we put in the design
# β[3] = whatever of x|x2 wasn't pivoted out, "multiply by 1" (ie do nothing)
#        because we had keep_coef = 1 in the design
@test only(eff.y) ≈ β[1] + 2 * β[2] + β[3]

# there is one bit of weirdness -- the removed coefficients behave like any other
# variable in the "design" that is missing from the model.
design = Dict(Symbol(dropped_coef) => [0])
eff_dropped = effects(design, model)

design = Dict(:q => [0])
eff_not_present = effects(design, model)

@test Matrix(eff_dropped) ≈ Matrix(eff_not_present)

# bootstrap
design = Dict(Symbol(kept_coef) => [1], :a => [2])
eff = effects(design, model)

boot = @suppress parametricbootstrap(StableRNG(666), 1000, model)
bootdf = unstack(DataFrame(boot.β), :coefname, :β)
# make sure that the bootstrap is correctly zeroing out
@test all(isequal(-0.0), bootdf[!, dropped_coef])

# now make sure the bootstrap gives approximately the same results
eff_boot = effects(design, model, boot)
@test eff_boot.y ≈ eff.y
@test eff_boot.err ≈ eff.err atol = 0.005

# test intervals and multiple rows in refgrid
design = Dict(Symbol(kept_coef) => 1:2, :a => [2])
eff95 = effects(design, model; level=0.95)
eff_boot95 = effects(design, model, boot; level=0.95)
@test eff_boot95.y ≈ eff95.y
@test eff_boot95.lower ≈ eff95.lower atol = 0.1
@test eff_boot95.upper ≈ eff95.upper atol = 0.1

glmm = @suppress fit(MixedModel,
                     @formula(use ~ 1 + age + abs2(age) + livch + urban + (1 | dist)),
                     MixedModels.dataset(:contra),
                     Bernoulli(); fast=true)

gboot = @suppress parametricbootstrap(StableRNG(345), 250, glmm)

design = Dict(:age => -1:1:1)
geff_boot = effects(design, glmm, gboot; eff_col="y", invlink=AutoInvLink(), level=0.95)
geff = effects(design, glmm; eff_col="y", invlink=AutoInvLink(), level=0.95)

@test geff_boot.y ≈ geff.y
@test geff_boot.err ≈ geff.err atol = 0.05
@test geff_boot.lower ≈ geff.lower atol = 0.05
@test geff_boot.upper ≈ geff.upper atol = 0.05
