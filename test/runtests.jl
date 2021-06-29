# using Distributions
using DataFrames
using Effects
using GLM
using StableRNGs
using StandardizedPredictors
using Statistics
using StatsModels
using Test

@testset "linear regression" begin
    b0, b1, b2, b1_2 = beta = [0.0, 1.0, 1.0, -1.0]

    @testset "simple" begin
        x = collect(-10:10)
        data = (; :x => x, :y => x .* b1 .+ b0 + randn(StableRNG(42), length(x)))
        model = lm(@formula(y ~ x), data)

        design = Dict(:x => 1:20)
        eff = effects(design, @formula(y ~ x), model)
        # test effect
        @test eff.y ≈ eff.x .* last(coef(model)) .+ first(coef(model))
        # test error
        pred = [1 first(eff.x)]
        # if we drop support for Julia < 1.4, this first can become only
        @test first(eff.err) ≈ first(sqrt(pred *  vcov(model) * pred'))
        # test CI
        @test eff.lower ≈ eff.y - eff.err
        @test eff.upper ≈ eff.y + eff.err

        @testset "contrasts" begin
            contrasts = Dict(:x => Center(10))
            model_centered = lm(@formula(y ~ x), data; contrasts=contrasts)

            eff_centered = effects(design, @formula(y ~ x), model_centered; contrasts=contrasts)

            # different contrasts shouldn't affect the predictions/effects
            @test eff_centered.y ≈ eff.y
            @test eff_centered.lower ≈ eff.lower
            @test eff_centered.upper ≈ eff.upper
        end
    end

    @testset "multiple" begin
        x = collect(-10:10)
        z = @. sqrt(x + 100)
        data = DataFrame(x=x, z=z, y=randn(StableRNG(42), length(x)))
        form = @formula(y ~ 1 + x * z)
        data[!, :y] += modelmatrix(form, data) * beta
        model = lm(form, data)

        design = Dict(:x => 1:20)
        maximin(v) = mean(extrema(v))
        for typical in (maximin, mean)
            eff = effects(design, @formula(y ~ x), model; typical=typical)
            # test effect
            z_typical = typical(z)
            zx_typical = z_typical .* typical(x)
            bs = coef(model)
            @test eff.y ≈ @. bs[1] + eff.x * bs[2] + bs[3] * z_typical + bs[4] * zx_typical
            # test error
            pred = [1 first(eff.x) z_typical zx_typical]
            # if we drop support for Julia < 1.4, this first can become only
            @test first(eff.err) ≈ first(sqrt(pred *  vcov(model) * pred'))
            # test CI
            @test eff.lower ≈ eff.y - eff.err
            @test eff.upper ≈ eff.y + eff.err
        end
    end
end
