using DataFrames
using Effects
using StatsModels
using MixedModels: MixedModel, ReMat
using Test

using Effects: get_matrix_term, typicalterm, typify, TypicalTerm
using Effects: _reference_grid

x = collect(-10:19)
dat = DataFrame(; x=x,
                z=repeat(["A", "B", "C"]; inner=10),
                y=zeros(length(x)))

@testset "get_matrix_term" begin
    form = @formula(y ~ 1 + x + (1 | z))
    f = apply_schema(form, schema(form, dat, Dict(:z => EffectsCoding())), MixedModel)
    rhs = f.rhs
    y, XZ = modelcols(f, dat)
    # test that we created an additional model matrix full of -0.0
    @test last(XZ) isa ReMat
    @test !(first(XZ) isa ReMat)
    mc = modelcols(get_matrix_term(rhs), dat)
    # test that the matrix term corresponds to the term we expect
    @test mc == first(XZ)
end

@testset "typicalterm" begin
    form = @formula(y ~ 1 + x * z)
    f = apply_schema(form, schema(form, dat, Dict(:z => EffectsCoding())))
    rhs = f.rhs
    y, X = modelcols(f, dat)
    @test_throws ArgumentError TypicalTerm(rhs.terms[1], [1, 2, 3, 4])

    # categorical var with fewer levels in refgrid as in original
    refgrid = _reference_grid(Dict(:x => [13.0, 15.0], :z => ["C"]))
    typicalf = typify(refgrid, f, X)
    # first column is the intercept
    # second column should be the x-values
    # third column is 0 because it's the "z: B" contrast and the mean of ±1 is 0
    # fourth column is 1 because it's "z: C" and we said that's the value we're taking
    # fifth col is the product of the cols 2&3
    # sixt col is the product of the cols 2&4
    @test modelcols(typicalf, refgrid) == Float64[1 13 0 1 0 13
                                                  1 15 0 1 0 15]

    # typical that isn't mean
    refgrid = _reference_grid(Dict(:z => ["A", "B"]))
    # typical should return a scalar
    @test_throws ArgumentError typify(refgrid, f, X; typical=extrema)
    typicalf = typify(refgrid, f, X; typical=minimum)
    typx = minimum(dat.x)
    # first col is intercept
    # second col is typx
    # third col is -1, 1 because it's the "z: B" contrast and we have 1 A and 1 B
    # fourth col is -1, 0 because it's the "z: C" contrast and we have 1 A and 1 B
    # fifth col is the product of the cols 2&3
    # sixt col is the product of the cols 2&4
    @test modelcols(typicalf, refgrid) == Float64[1 typx -1 -1 -typx  -typx
                                                  1 typx  1  0  typx      0]
    # typical of categorical vars
    f = apply_schema(form, schema(form, dat, Dict(:z => DummyCoding())))
    rhs = f.rhs
    y, X = modelcols(f, dat)
    refgrid = _reference_grid(Dict(:x => [π]))
    typicalf = typify(refgrid, f, X)
    # first col is intercept
    # second col is 3.14
    # third col is 1/3 because it's a balanced design
    # fourth col is 1/3 because it's a balanced design
    # fifth col is the product of the cols 2&3
    # sixt col is the product of the cols 2&4
    @test modelcols(typicalf, refgrid) ≈ Float64[1 π 1/3 1/3 π/3 π/3]

    # weird models
    form = @formula(y ~ 0 + x + x & z)
    f = apply_schema(form, schema(form, dat))
    rhs = f.rhs
    y, X = modelcols(f, dat)
    refgrid = _reference_grid(Dict(:x => [π]))
    @test_throws ArgumentError typify(refgrid, f, X)
end
