using DataFrames
using Effects
using GLM
using StableRNGs
using StandardizedPredictors
using Statistics
using StatsModels
using Test

@testset "TypicalTerm" begin
    include("typical.jl")
end

@testset "linear regression" begin

    b0, b1, b2, b1_2 = beta = [0.0, 1.0, 1.0, -1.0]

    @testset "simple" begin
        x = collect(-10:10)
        dat = (; :x => x, :y => x .* b1 .+ b0 + randn(StableRNG(42), length(x)))
        model = lm(@formula(y ~ x), dat)

        design = Dict(:x => 1:20)
        eff = effects(design, model)
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
            model_centered = lm(@formula(y ~ x), dat; contrasts=contrasts)

            eff_centered = effects(design, model_centered)

            # different contrasts shouldn't affect the predictions/effects
            @test eff_centered.y ≈ eff.y
            @test eff_centered.lower ≈ eff.lower
            @test eff_centered.upper ≈ eff.upper
        end

    end

    @testset "multiple" begin

    end
end
