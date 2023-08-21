using DataFrames
using Effects
using GLM
using LinearAlgebra
using MixedModels
using RDatasets: dataset as rdataset
using StableRNGs
using Test

@testset "transformed response" begin
    dat = rdataset("car", "Prestige")
    design = Dict(:Income => [1, 2],
                  :Education => [3, 4])
    model = lm(@formula(log(Prestige) ~ 1 + Income * Education), dat)
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

    @testset "AutoInvLink fails gracefully" begin
        # this should work even pre Julia 1.9 because by definition
        # no extension is loaded
        @test_throws ArgumentError effects(design, model; invlink=AutoInvLink())
    end
end

@testset "link function" begin
    dat = rdataset("car", "Cowles")
    dat[!, :vol] = dat.Volunteer .== "yes"
    model = glm(@formula(vol ~ Extraversion * Neuroticism), dat, Bernoulli())
    design = Dict(:Extraversion => [13],
                  :Neuroticism => [16])
    X = [1.0 13.0 16.0 13 * 16]
    iv = Base.Fix1(GLM.linkinv, Link(model.model))
    @static if VERSION >= v"1.9"
        invlinks = [iv, AutoInvLink()]
        @test Effects._model_link(model, AutoInvLink()) ==
              Effects._model_link(model.model, AutoInvLink())
    else
        invlinks = [iv]
    end
    @testset "invlink = $invlink" for invlink in invlinks
        for level in [0.68, 0.95]
            eff = effects(design, model; invlink, level)

            # compare with results from GLM.predict
            pred = DataFrame(predict(model.model, X;
                                     interval=:confidence,
                                     interval_method=:delta,
                                     level))
            @test all(pred.prediction .≈ eff.vol)
            @test all(isapprox.(pred.lower, eff.lower; atol=0.001))
            @test all(isapprox.(pred.upper, eff.upper; atol=0.001))

            eff_trans = effects(design, model; level)
            transform!(eff_trans,
                       :vol => ByRow(iv),
                       :lower => ByRow(iv),
                       :upper => ByRow(iv); renamecols=false)
            # for this model, things play out nicely
            @test all(eff_trans.vol .≈ eff.vol)
            @test all(isapprox.(eff_trans.lower, eff.lower; atol=0.001))
            @test all(isapprox.(eff_trans.upper, eff.upper; atol=0.001))
        end

        # compare with results from emmeans in R
        # emmeans(model, ~ neuroticism * extraversion, level=0.68)
        eff_emm = effects(Dict(:Extraversion => [12.4], :Neuroticism => [11.5]), model)
        @test isapprox(only(eff_emm.vol), -0.347; atol=0.005)
        @test isapprox(only(eff_emm.err), 0.0549; atol=0.005)
        @test isapprox(only(eff_emm.lower), -0.402; atol=0.005)
        @test isapprox(only(eff_emm.upper), -0.292; atol=0.005)

        # emmeans(model, ~ neuroticism * extraversion, level=0.68, transform="response")
        eff_emm_trans = effects(Dict(:Extraversion => [12.4], :Neuroticism => [11.5]),
                                model;
                                invlink)
        @test isapprox(only(eff_emm_trans.vol), 0.414; atol=0.005)
        @test isapprox(only(eff_emm_trans.err), 0.0133; atol=0.005)
        @test isapprox(only(eff_emm_trans.lower), 0.401; atol=0.005)
        @test isapprox(only(eff_emm_trans.upper), 0.427; atol=0.005)
    end
end

@static if VERSION >= v"1.9"
    @testset "Non Link01 GLM link" begin
        dat = rdataset("car", "Cowles")
        dat[!, :vol] = dat.Volunteer .== "yes"
        # this isn't a particularly sensible model, but it's fine for testing
        model = glm(@formula(vol ~ Extraversion * Neuroticism), dat, Poisson())
        design = Dict(:Extraversion => [13],
                      :Neuroticism => [16])
        X = [1.0 13.0 16.0 13 * 16]
        eff_manual = effects(design, model;
                             invlink=Base.Fix1(GLM.linkinv, Link(model.model)))
        eff_auto = effects(design, model; invlink=AutoInvLink())

        @test all(isapprox.(Matrix(eff_manual), Matrix(eff_auto)))
    end
    @testset "link function in a MixedModel" begin
        model = fit(MixedModel,
                    @formula(use ~ 1 + age + (1 | urban)),
                    MixedModels.dataset(:contra),
                    Bernoulli(); progress=false)
        design = Dict(:age => -10:10)
        eff_manual = effects(design, model;
                             invlink=Base.Fix1(GLM.linkinv, Link(model)))
        eff_auto = effects(design, model; invlink=AutoInvLink())

        @test all(isapprox.(Matrix(eff_manual), Matrix(eff_auto)))
    end
end

@testset "identity by another name" begin
    b0, b1, b2, b1_2 = beta = [0.0, 1.0, 1.0, -1.0]
    x = collect(-10:10)
    dat = (; :x => x, :y => x .* b1 .+ b0 + randn(StableRNG(42), length(x)))
    model = lm(@formula(y ~ x), dat)

    invlink = x -> x # same as identity but won't trigger that branch
    @test invlink !== identity

    design = Dict(:x => 1:20)
    # if our math on the delta method is correct, then it should work for the
    # identity case, even though we special case identity() to reduce the
    # computation
    eff = effects(design, model; invlink=identity)
    eff_link = effects(design, model; invlink)
    # these should be exactly equal b/c derivative is just a bunch of ones
    # however, we may have to loosen this to approximate equality if
    # the linear algebra gets very optimized and we start seeing the effects
    # of associativity in SIMD operations
    @test eff == eff_link
end
