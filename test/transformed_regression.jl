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
end

@testset "link function" begin
    dat = rdataset("car", "Cowles")
    dat[!, :vol] = dat.Volunteer .== "yes"
    model = glm(@formula(vol ~ Extraversion * Neuroticism), dat, Bernoulli())
    invlink = Base.Fix1(GLM.linkinv, Link(model.model))
    design = Dict(:Extraversion => [13],
                  :Neuroticism => [16])
    eff = effects(design, model; invlink)
    X = [1.0 13.0 16.0 13*16]
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
end
