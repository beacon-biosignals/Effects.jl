#=

Idea is: we have a model formula and a design dict of variables with reference values.
all teh terms that are not present in the design get "typified".  we re-write the model
formula replacing all the terms with their "typified" version where available.

then we call modelcols using this new formula + the reference grid

=#

struct TypicalTerm{T,V} <: AbstractTerm
    term::T
    values::V
    function TypicalTerm(t::T, v::V) where {T<:AbstractTerm,V}
        n = length(v)
        m = width(t)
        if n != m
            throw(ArgumentError("mismatch between number of typical values ($n) " *
                                "and number of columns generated by term $t ($m)"))
        end
        return new{T,V}(t, v)
    end
end

function StatsModels.modelcols(t::TypicalTerm, d::NamedTuple)
    cols = ones(length(first(d)), width(t))
    for (idx, v) in enumerate(t.values)
        cols[:, idx] .*= v
    end
    return cols
end

StatsBase.coefnames(t::TypicalTerm) = coefnames(t.term)
# regular show
Base.show(io::IO, t::TypicalTerm) = show(io, t.term)
# long show
Base.show(io::IO, m::MIME"text/plain", t::TypicalTerm) = show(io, m, t.term)

# statsmodels glue code:
StatsModels.width(t::TypicalTerm) = StatsModels.width(t.term)
# don't generate schema entries for terms which are already typified
StatsModels.needs_schema(::TypicalTerm) = false
StatsModels.termsyms(t::TypicalTerm) = StatsModels.termsyms(t.term)

function get_matrix_term(x)
    x = StatsModels.collect_matrix_terms(x)
    x = x isa MatrixTerm ? x : first(x)
    x isa MatrixTerm || throw(ArgumentError("couldn't extract matrix term from $x"))
    if first(x.terms) isa MatrixTerm
        x = only(x.terms)
    end
    return x
end

function typify(refgrid, model_formula::FormulaTerm,
                model_matrix::AbstractMatrix; typical=mean)
    effects_terms = Term.(Tables.columnnames(refgrid))

    # creates a MatrixTerm (and should work for things like MixedModels) which
    # should correspond to the model_matrix
    matrix_term = get_matrix_term(model_formula.rhs)
    typical_terms = Dict()

    urterms = terms(matrix_term)
    nonfunc_terms = [tt for tt in matrix_term.terms if !isa(tt, FunctionTerm)]
    # only include generating terms that exist outside of a FunctionTerm
    filter!(urterms) do tt
        return any(x -> _termsyms(tt) < _termsyms(x), nonfunc_terms)
    end

    # we need to include FunctionTerms separately so that we can grab the transformed columns
    func_terms = [filter(tt -> isa(tt, FunctionTerm), matrix_term.terms)...]
    for term in [urterms; func_terms]
        if !any(et -> _symequal(et, term), effects_terms)
            typical_terms[term] = typicalterm(term, matrix_term, model_matrix; typical=typical)
        end
    end

    return _replace(matrix_term, typical_terms)
end

_replace(matrix_term::MatrixTerm, typicals::Dict) = MatrixTerm(_replace.(matrix_term.terms, Ref(typicals)))
_replace(term::AbstractTerm, typicals::Dict) = haskey(typicals, term) ? typicals[term] : term
_replace(term::InteractionTerm, typicals::Dict) = InteractionTerm(_replace.(term.terms, Ref(typicals)))

_rmliterals(s) = filter(x -> !isa(x, Number), s)
_termsyms(t) = StatsModels.termsyms(t)
_termsyms(t::FunctionTerm{Fo, Fa, Names}) where {Fo, Fa, Names} = Names
_symequal(t1::AbstractTerm, t2::AbstractTerm) = issetequal(_termsyms(t1), _termsyms(t2))

_trmequal(t1::AbstractTerm, t2::AbstractTerm) = _symequal(t1, t2)
_trmequal(t1::AbstractTerm, t2::FunctionTerm) = false
_trmequal(t1::FunctionTerm, t2::AbstractTerm) = false
function _trmequal(t1::FunctionTerm, t2::FunctionTerm)
    return t1.exorig == t2.exorig &&
        _symequal(t1, t2) &&
        t1.forig == t2.forig
end

function typicalterm(term::AbstractTerm, context::MatrixTerm, model_matrix; typical=mean)
    i = findfirst(t -> _trmequal(t, term), context.terms)
    i === nothing && throw(ArgumentError("Can't determine columns corresponding to '$term' in matrix term $context"))
    cols = (i == 1 ? 0 : sum(width, context.terms[1:(i - 1)])) .+ (1:width(term))
    vals = map(typical, eachcol(view(model_matrix, :, cols)))
    all(v -> length(v) == 1, vals) || throw(ArgumentError("Typical function should return a scalar."))
    return TypicalTerm(term, vals)
end
