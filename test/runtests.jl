using Aqua
using Effects
using Test
using TestSetExtensions

@testset ExtendedTestSet "Effects.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(Effects; ambiguities=false)
    end

    @testset "TypicalTerm" begin
        include("typical.jl")
    end

    @testset "linear regression" begin
        include("linear_regression.jl")
    end

    @testset "delta method" begin
        include("delta_method.jl")
    end

    @testset "emmeans" begin
        include("emmeans.jl")
    end

    @testset "MixedModels.jl" begin
        include("mixedmodels.jl")
    end
end
