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
        # waiting on https://github.com/JuliaStats/StatsModels.jl/pull/211
        effects(design, @formula(y ~ x), model)
    end

    @testset "multiple" begin

    end
end
