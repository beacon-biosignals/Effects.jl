using Test

@testset "TypicalTerm" begin
    include("typical.jl")
end

@testset "linear regression" begin
    include("linear_regression.jl")
end

@testset "delta method" begin
    include("delta_method.jl")
end
