using DataFrames
using Effects
using GLM
using LinearAlgebra
using StableRNGs
using StandardizedPredictors
using Statistics
using StatsModels
using Test

rng = StableRNG(42)
growthdata = DataFrame(; age=[13:20; 13:20],
                       sex=repeat(["male", "female"], inner=8),
                       weight=[range(100, 155; length=8); range(100, 125; length=8)] .+ randn(rng, 16))
# now make unbalanced
growthdata = growthdata[1:(end-2), :]
model = lm(@formula(weight ~ 1 + sex * age), growthdata)

# values from R
# R> em <- emmeans(model, ~ 1 + sex * age)
# R> summary(em)$emmean
# [1] 110.8796 124.3732
# R> summary(em)$SE
# [1] 0.4741895 0.3961926

em = emmeans(model)
@test all(isapprox.(em.weight, [110.8796, 124.3732]; atol=0.001))
@test all(isapprox.(em.err, [0.4741895. 0.3961926]; atol=0.001))

# emmeans(model; levels=Dict(:age => 23))

# R> summary(pairs(em))$estimate
# [1] -13.49363
# R> summary(pairs(em))$SE
# [1] 0.6179193

emp = empairs(model)
@test only(emp.weight) ≈ -13.49363 atol=0.001
@test only(emp.err) ≈ 0.6179193 atol=0.001
