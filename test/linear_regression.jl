using DataFrames
using Effects
using GLM
using LinearAlgebra
using StableRNGs
using StandardizedPredictors
using Statistics
using StatsModels
using Test
using Vcov

using StatsAPI: crossmodelmatrix

b0, b1, b2, b1_2 = beta = [0.0, 1.0, 1.0, -1.0]

# test piracy for compat with later versions of Vcov.jl
@static if !hasmethod(Vcov.invcrossmodelmatrix, Tuple{RegressionModel})
    Vcov.invcrossmodelmatrix(x::RegressionModel) = inv(crossmodelmatrix(x))
end

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
    @test first(eff.err) ≈ only(sqrt(pred * vcov(model) * pred'))
    # test CI
    @test eff.lower ≈ eff.y - eff.err
    @test eff.upper ≈ eff.y + eff.err

    @testset "custom vcov" begin
        eff2 = effects(design, model; vcov=x -> 2 * vcov(x))
        @test first(eff2.err) ≈ only(sqrt(pred * 2vcov(model) * pred'))
        # table regression model *grumble*
        effrob = effects(design, model; vcov=m -> vcov(m.model, Vcov.robust()))
        # robust errors are larger
        @test effrob.err > eff.err
    end

    @testset "contrasts" begin
        # need to check this works when the response has also
        # had a name change
        contrasts = Dict(:x => Center(10), :y => ZScore())
        model_centered = lm(@formula(y ~ x), dat; contrasts=contrasts)

        eff_centered = effects(design, model_centered)

        μ = mean(dat.y)
        σ = std(dat.y)
        # different contrasts shouldn't affect the predictions/effects
        @test eff_centered[!, "y(centered: 0.21 scaled: 6.27)"] ≈ zscore(eff.y, µ, σ)
        @test eff_centered.lower ≈ zscore(eff.lower, µ, σ)
        @test eff_centered.upper ≈ zscore(eff.upper, µ, σ)
    end
end

@testset "multiple" begin
    x = collect(-10:10)
    z = @. sqrt(x + 100)
    dat = DataFrame(; x=x, z=z, y=randn(StableRNG(42), length(x)))

    @testset "additive" begin
        form = @formula(y ~ 1 + x + z)
        dat[!, :y] += modelmatrix(form, dat) * beta[1:(end - 1)]
        model = lm(form, dat)

        design = Dict(:x => 1:20)
        maximin(v) = mean(extrema(v))
        for typical in (maximin, mean)
            eff = effects(design, model; typical=typical)
            # test effect
            intercept = ones(length(design[:x]))
            z_typical = typical(z) * intercept
            bs = coef(model)
            @test eff.y ≈ @. bs[1] + eff.x * bs[2] + bs[3] * z_typical
            # test error
            pred = [intercept eff.x z_typical]
            @test eff.err ≈ sqrt.(diag(pred * vcov(model) * pred'))
            # test CI
            @test eff.lower ≈ eff.y - eff.err
            @test eff.upper ≈ eff.y + eff.err
        end
    end

    @testset "multiplicative" begin
        form = @formula(y ~ 1 + x * z)
        dat[!, :y] += modelmatrix(form, dat) * beta
        model = lm(form, dat)

        design = Dict(:x => 1:20)
        maximin(v) = mean(extrema(v))
        for typical in (maximin, mean)
            eff = effects(design, model; typical=typical)
            # test effect
            intercept = ones(length(design[:x]))
            z_typical = typical(z) * intercept
            zx_typical = z_typical .* design[:x]
            bs = coef(model)
            @test eff.y ≈ @. bs[1] + eff.x * bs[2] + bs[3] * z_typical + bs[4] * zx_typical
            # test error
            pred = [intercept eff.x z_typical zx_typical]
            @test eff.err ≈ sqrt.(diag(pred * vcov(model) * pred'))
            # test CI
            @test eff.lower ≈ eff.y - eff.err
            @test eff.upper ≈ eff.y + eff.err
        end
    end
end
