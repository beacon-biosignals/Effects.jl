# using Distributions
using DataFrames
using Effects
using GLM
using StableRNGs
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
    end

    @testset "multiple" begin

    end
end
