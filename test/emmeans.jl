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
growthdata[(end-2):end, :sex] .= "other"
model = lm(@formula(weight ~ 1 + sex * age), growthdata)

# values from R
# R> em <- emmeans(model, ~ 1 + sex * age)

# R> em
#  sex     age emmean    SE df lower.CL upper.CL
#  female 16.5    112 0.706 10      110      113
#  male   16.5    128 0.383 10      127      129
#  other  16.5    113 2.014 10      109      118

# Confidence level used: 0.95

# R> summary(em)$emmean
# [1] 111.6818 127.6822 113.1625


# R> summary(em)$SE
# [1] 0.7060401 0.3829040 2.0140411

em = emmeans(model)
@test all(isapprox.(em.weight, [111.6818, 127.6822, 113.1625]; atol=0.001))
@test all(isapprox.(em.err, [0.7060401, 0.3829040, 2.0140411]; atol=0.001))

# emmeans(model; levels=Dict(:age => 23))

# R> pairs(em)
#  contrast                       estimate    SE df t.ratio p.value
#  female age16.5 - male age16.5    -16.00 0.803 10 -19.921  <.0001
#  female age16.5 - other age16.5    -1.48 2.134 10  -0.694  0.7724
#  male age16.5 - other age16.5      14.52 2.050 10   7.082  0.0001

# P value adjustment: tukey method for comparing a family of 3 estimates

# R> summary(pairs(em))$estimate
# [1] -16.000323  -1.480698  14.519625

# R> summary(pairs(em))$SE
# [1] 0.8031862 2.1342104 2.0501163

emp = empairs(model)
@test names(emp) == ["sex", "age", "weight", "err"]
@test emp.sex == ["female > male", "female > other", "male > other"]
@test all(emp.age .== 16.5)
@test all(isapprox.(emp.weight, [-16.000323, -1.480698, 14.519625]; atol=0.001))
@test all(isapprox.(emp.err, [0.8031862, 2.1342104, 2.0501163]; atol=0.001))
