using Aqua
using Effects
using Test

@testset "Aqua" begin
    # layout of weakdeps and extensions etc. differs between pre 1.9 and 1.9+
    project_toml_formatting = VERSION >= v"1.9"
    Aqua.test_all(Effects; ambiguities=false, project_toml_formatting)
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
