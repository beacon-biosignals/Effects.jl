using Aqua
using Effects
using Test

@testset "Aqua" begin
    Aqua.test_all(Effects; ambiguities=false)
end

@testset "TypicalTerm" include("typical.jl")

@testset "linear regression" include("linear_regression.jl")

@testset "delta method" include("delta_method.jl")

@testset "emmeans" include("emmeans.jl")
