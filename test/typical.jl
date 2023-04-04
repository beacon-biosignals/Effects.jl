using DataFrames
using Effects
using StatsBase
using StatsModels
using MixedModels: MixedModel, ReMat
using Test

using DataStructures: DefaultDict
using Effects: get_matrix_term, typicalterm, typify, TypicalTerm

x = collect(-10:19)
dat = DataFrame(; x=x,
                w=exp.(x),
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

    trm = first(form.rhs)
    materm = get_matrix_term(rhs)
    mc = modelcols(materm, dat)
    # this tests that we're fully extracting the nested MatrixTerm
    @test typicalterm(trm, materm, mc) isa TypicalTerm
    # test that the matrix term corresponds to the term we expect
    @test mc == first(XZ)
end

@testset "typicalterm" begin
    form = @formula(y ~ 1 + x * z)
    f = apply_schema(form, schema(form, dat, Dict(:z => EffectsCoding())))
    rhs = f.rhs
    y, X = modelcols(f, dat)
    @test_throws ArgumentError TypicalTerm(rhs.terms[1], [1, 2, 3, 4])

    @test !StatsModels.needs_schema(TypicalTerm(rhs.terms[1], 1))

    for term in rhs.terms
        w = width(term)
        typ = TypicalTerm(term, ones(w))

        @test width(typ) == w
        @test coefnames(typ) == coefnames(term)
        @test sprint(show, typ) == sprint(show, term)
        mime = MIME("text/plain")
        @test sprint(show, mime, typ) == sprint(show, mime, term)
        @test StatsModels.termsyms(typ) == StatsModels.termsyms(term)
    end

    # categorical var with fewer levels in refgrid as in original
    refgrid = expand_grid(Dict(:x => [13.0, 15.0], :z => ["C"]))
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
    refgrid = expand_grid(Dict(:z => ["A", "B"]))
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
    @test modelcols(typicalf, refgrid) == Float64[1 typx -1 -1 -typx -typx
                                                  1 typx 1 0 typx 0]
    # typical of categorical vars
    f = apply_schema(form, schema(form, dat, Dict(:z => DummyCoding())))
    rhs = f.rhs
    y, X = modelcols(f, dat)
    refgrid = expand_grid(Dict(:x => [π]))
    typicalf = typify(refgrid, f, X)
    # first col is intercept
    # second col is 3.14
    # third col is 1/3 because it's a balanced design
    # fourth col is 1/3 because it's a balanced design
    # fifth col is the product of the cols 2&3
    # sixt col is the product of the cols 2&4
    @test modelcols(typicalf, refgrid) ≈ Float64[1 π 1 / 3 1 / 3 π / 3 π / 3]

    # weird models
    form = @formula(y ~ 0 + x + x & z)
    f = apply_schema(form, schema(form, dat))
    rhs = f.rhs
    y, X = modelcols(f, dat)
    refgrid = expand_grid(Dict(:x => [π]))
    @test_throws ArgumentError typify(refgrid, f, X)

    @testset "dictionary typicals" begin
        typical_fun = maximum
        # need to wrap this in a lambda because functions are treating specially
        typical_dict = DefaultDict(() -> typical_fun)

        form = @formula(y ~ 1 + x * z)
        f = apply_schema(form, schema(form, dat, Dict(:z => EffectsCoding())))
        rhs = f.rhs
        y, X = modelcols(f, dat)
        refgrid = expand_grid(Dict(:z => ["A", "B"]))
        # since we're not specifying anything in the default dict, we always
        # hit the default and these should give the same result
        tf = modelcols(typify(refgrid, f, X; typical=typical_fun), refgrid)
        td = modelcols(typify(refgrid, f, X; typical=typical_dict), refgrid)
        @test tf == td

        form = @formula(y ~ 1 + w + x + z)
        f = apply_schema(form, schema(form, dat))
        rhs = f.rhs
        y, X = modelcols(f, dat)
        typical_dict[:w] = minimum
        td = modelcols(typify(refgrid, f, X; typical=typical_dict), refgrid)
        minw = minimum(dat.w)
        maxx = maximum(dat.x)
        # only checking the first level
        expected = [1.0 minw maxx 0.0 0.0
                    1.0 minw maxx 1.0 0.0]
        @test td == expected

        # without using a default dict, make sure we don't need
        # to specify the intercept
        form = @formula(y ~ 1 + x + z)
        f = apply_schema(form, schema(form, dat))
        rhs = f.rhs
        y, X = modelcols(f, dat)
        td = modelcols(typify(refgrid, f, X; typical=Dict(:x => mean)), refgrid)
        meanx = mean(dat.x)
        @test td == [1.0 meanx 0.0 0.0
                     1.0 meanx 1.0 0.0]

        # this should error on :x, not :1
        @test_throws KeyError(:x) typify(refgrid, f, X; typical=Dict())
    end
end

@testset "_trmequal" begin
    # get that 100% coverage
    form = @formula(y ~ 0 + x + x^2 + x^3)
    f = apply_schema(form, schema(form, dat))
    terms = f.rhs.terms
    # terms[3] is the quadratic FunctionTerm
    # terms[2] is the linear Continuous
    @test !Effects._trmequal(terms[3], terms[2])
    @test !Effects._trmequal(terms[2], terms[3])
    @test Effects._trmequal(terms[2], terms[2])
    @test Effects._trmequal(terms[3], terms[3])
end

@testset "FunctionTerm" begin
    # no untransformed w here -- we want to make sure we don't try
    # to grab the nonexistent column corresponding to untransformed w
    form = @formula(y ~ 0 + x + log(w) + sqrt(w))
    f = apply_schema(form, schema(form, dat))
    rhs = f.rhs
    X = modelcols(rhs, dat)
    refgrid = expand_grid(Dict(:x => [π]))
    @test modelcols(typify(refgrid, f, X), refgrid) ≈
          Float64[π mean(log.(dat.w)) mean(sqrt.(dat.w))]
    refgrid = expand_grid(Dict(:x => [π], :w => [π]))
    @test modelcols(typify(refgrid, f, X), refgrid) ≈ Float64[π log(π) sqrt(π)]

    form = @formula(y ~ 0 + x + x^2 + x^3)
    f = apply_schema(form, schema(form, dat))
    rhs = f.rhs
    X = modelcols(rhs, dat)
    refgrid = expand_grid(Dict(:x => [π]))
    @test modelcols(typify(refgrid, f, X), refgrid) ≈ Float64[π π^2 π^3]
end
