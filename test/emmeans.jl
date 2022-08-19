using DataFrames
using Effects
using GLM
using MultipleTesting
using StableRNGs
using StandardizedPredictors
using Statistics
using StatsModels
using Test

rng = StableRNG(42)
growthdata = DataFrame(;
                       age=[13:20; 13:20],
                       sex=repeat(["male", "female"]; inner=8),
                       weight=[range(100, 155; length=8); range(100, 125; length=8)] .+
                              randn(rng, 16))
# now make unbalanced
growthdata[(end - 2):end, :sex] .= "other"
model = lm(@formula(weight ~ 1 + sex * age), growthdata)
model_scaled = lm(@formula(weight ~ 1 + sex * age), growthdata;
                  contrasts=Dict(:age => ZScore(), :sex => EffectsCoding()))
# values from R
# done using default contrasts and without scaling
# which is a good test that these things are contrast invariant
# (like they're supposed to be)
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
@testset "emmeans" for m in [model, model_scaled]
    em = emmeans(m)
    @test all(em.age .== 16.5)
    @test all(isapprox.(em.weight, [111.6818, 127.6822, 113.1625]; atol=0.001))
    @test all(isapprox.(em.err, [0.7060401, 0.3829040, 2.0140411]; atol=0.001))
    @test !in("dof", names(em))

    @testset "dof" begin
        em = emmeans(m; dof=dof_residual)
        @test all(em.dof .== 10)
    end
    em = emmeans(m; levels=Dict(:age => 23))
    @test all(em.age .== 23)
end

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

# R> summary(pairs(em, adjust=NULL))$p.value
# [1] 2.230652e-09 5.036095e-01 3.365674e-05

# R> summary(pairs(em, adjust="bonferroni"))$p.value
# [1] 6.691955e-09 1.000000e+00 1.009702e-04

bonferroni(pvals) = adjust(PValues(pvals), Bonferroni())

@testset "empairs" for m in [model, model_scaled]
    emp = empairs(m)
    @test names(emp) == ["sex", "age", "weight", "err"]
    @test emp.sex == ["female > male", "female > other", "male > other"]
    @test all(emp.age .== 16.5)
    @test all(isapprox.(emp.weight, [-16.000323, -1.480698, 14.519625]; atol=0.001))
    @test all(isapprox.(emp.err, [0.8031862, 2.1342104, 2.0501163]; atol=0.001))
    @test !in("dof", names(emp))

    @testset "dof" begin
        emp = empairs(m; dof=Inf)
        @test all(emp.dof .== Inf)

        emp = empairs(m; dof=dof_residual)
        @test all(emp.dof .== 10)
        @test all(isapprox.(emp.t, [-19.921, -0.694, 7.082]; atol=0.001))
        @test all(isapprox.(emp[!, "Pr(>|t|)"], [2.230652e-09, 5.036095e-01, 3.365674e-05];
                            rtol=0.001))

        emp_adj = empairs(m; dof=dof_residual, padjust=bonferroni)
        @test all(isapprox.(emp_adj[!, "Pr(>|t|)"], [6.691955e-09, 1.0, 1.009702e-04];
                            rtol=0.001))
    end
end
