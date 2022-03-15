using DataFrames
using Effects
using GLM
using LinearAlgebra
using RDatasets: dataset as rdataset
using Test

@testset "transformed response" begin
    dat = rdataset("car", "Prestige")
    model = lm(@formula(log(Prestige) ~ 1 + Income * Education), dat)
    design = Dict(:Income => [1, 2],
                  :Education => [3, 4])
    eff_original_scale = effects(design, model; invlink=exp)
    eff_logscale = effects(design, model)
    @test all(eff_logscale[!, 3] .≈ log.(eff_original_scale[!, 3]))
    # the derivative of the exponential function is the exponential function....
    deriv = exp.(eff_logscale[!, 3])
    err = eff_logscale.err .* deriv
    @test all(eff_original_scale.err .≈ err)

    # compare with results from emmeans in R
    # relatively high tolerances for the point estimates b/c that's very susceptible
    # to variation in the coef estimates, but the vcov estimates are more stable
    # emmeans(model,  ~ income * education, level=0.68)
    eff_emm = effects(Dict(:Income => [6798], :Education => [10.7]), model;
                      eff_col="log(Prestige)")
    @test isapprox(only(eff_emm[!, "log(Prestige)"]), 3.84; atol=0.01)
    @test isapprox(only(eff_emm.err), 0.023; atol=0.005)
    @test isapprox(only(eff_emm.lower), 3.81; atol=0.005)
    @test isapprox(only(eff_emm.upper), 3.86; atol=0.005)

    # emmeans(model, ~ income * education, level=0.68, transform="response")
    eff_emm_trans = effects(Dict(:Extraversion => [12.4], :Neuroticism => [11.5]), model;
                            invlink=exp, eff_col="Prestige")
    @test isapprox(only(eff_emm_trans[!, "Prestige"]), 46.4; atol=0.05)
    @test isapprox(only(eff_emm_trans.err), 1.07; atol=0.005)
    @test isapprox(only(eff_emm_trans.lower), 45.3; atol=0.05)
    @test isapprox(only(eff_emm_trans.upper), 47.5; atol=0.05)
end

@testset "link function" begin
    dat = rdataset("car", "Cowles")
    dat[!, :vol] = dat.Volunteer .== "yes"
    model = glm(@formula(vol ~ Extraversion * Neuroticism), dat, Bernoulli())
    invlink = Base.Fix1(GLM.linkinv, Link(model.model))
    design = Dict(:Extraversion => [13],
                  :Neuroticism => [16])
    eff = effects(design, model; invlink)
    X = [1.0 13.0 16.0 13 * 16]
    # compare with results from GLM.predict
    pred = DataFrame(predict(model.model, X;
                             interval=:confidence,
                             interval_method=:delta,
                             level=0.68)) # 0.68 is 1 normal quantile, which is just the SE
    @test all(pred.prediction .≈ eff.vol)
    @test all(isapprox.(pred.lower, eff.lower; atol=0.001))
    @test all(isapprox.(pred.upper, eff.upper; atol=0.001))

    eff_trans = effects(design, model)
    transform!(eff_trans,
               :vol => ByRow(invlink),
               :lower => ByRow(invlink),
               :upper => ByRow(invlink); renamecols=false)
    # for this model, things play out nicely
    @test all(eff_trans.vol .≈ eff.vol)
    @test all(isapprox.(eff_trans.lower, eff.lower; atol=0.001))
    @test all(isapprox.(eff_trans.upper, eff.upper; atol=0.001))

    # compare with results from emmeans in R
    # emmeans(model, ~ neuroticism * extraversion, level=0.68)
    eff_emm = effects(Dict(:Extraversion => [12.4], :Neuroticism => [11.5]), model)
    @test isapprox(only(eff_emm.vol), -0.347; atol=0.005)
    @test isapprox(only(eff_emm.err), 0.0549; atol=0.005)
    @test isapprox(only(eff_emm.lower), -0.402; atol=0.005)
    @test isapprox(only(eff_emm.upper), -0.292; atol=0.005)

    # emmeans(model, ~ neuroticism * extraversion, level=0.68, transform="response")
    eff_emm_trans = effects(Dict(:Extraversion => [12.4], :Neuroticism => [11.5]), model;
                            invlink)
    @test isapprox(only(eff_emm_trans.vol), 0.414; atol=0.005)
    @test isapprox(only(eff_emm_trans.err), 0.0133; atol=0.005)
    @test isapprox(only(eff_emm_trans.lower), 0.401; atol=0.005)
    @test isapprox(only(eff_emm_trans.upper), 0.427; atol=0.005)
end
