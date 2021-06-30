using DataFrames
using Effects
using StatsModels
using Test

using Effects: get_matrix_term, typicalterm, typify, TypicalTerm

include("reterm.jl")

x = collect(-10:9)
dat = DataFrame(; x=x,
                z=repeat(["A", "B"]; inner=10),
                y=zeros(length(x)))


form = @formula(y ~ x * z)
f = apply_schema(form, schema(form, dat, Dict(:z => EffectsCoding())))

rhs = f.rhs
y, X = modelcols(f, dat)

@testset "typicalterm" begin
    @test_throws ArgumentError TypicalTerm(rhs.terms[1], [1,2,3,4])
end

@testset "get_matrix_term" begin
    form = @formula(y ~ 1 + x + (1|z))
    f = apply_schema(form, schema(form, dat, Dict(:z => EffectsCoding())), FakeMixed)
    rhs = f.rhs
    y, XZ = modelcols(f, dat)
    # test that we created an additional model matrix full of -0.0
    @test all(signbit, last(XZ))
    mc = modelcols(get_matrix_term(rhs), dat)
    # test that the matrix term corresponds to the term we expect
    @test mc == first(XZ)
    # test that the matrix term isn't all -0.0s like the re term
    @test any(>(0), mc)
end
