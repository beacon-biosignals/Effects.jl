# this is adapated from MixedModels.jl @9d2b965
# easier to include this one file for testing than to
# add a heavyweight testing dependency

using StatsModels
using StatsModels: Schema

abstract type FakeMixed end


struct RandomEffectsTerm <: AbstractTerm
    lhs::StatsModels.TermOrTerms
    rhs::StatsModels.TermOrTerms
end

# TODO: consider overwriting | with our own function that can be
# imported with (a la FilePathsBase.:/)
# using MixedModels: |
# to avoid conflicts with definitions in other packages...
Base.:|(a::StatsModels.TermOrTerms, b::StatsModels.TermOrTerms) = RandomEffectsTerm(a, b)

# expand (lhs | a + b) to (lhs | a) + (lhs | b)
RandomEffectsTerm(lhs::StatsModels.TermOrTerms, rhs::NTuple{2,AbstractTerm}) =
    (RandomEffectsTerm(lhs, rhs[1]), RandomEffectsTerm(lhs, rhs[2]))

Base.show(io::IO, t::RandomEffectsTerm) = Base.show(io, MIME"text/plain"(), t)

Base.show(io::IO, ::MIME"text/plain", t::RandomEffectsTerm) = print(io, "($(t.lhs) | $(t.rhs))")
StatsModels.is_matrix_term(::Type{RandomEffectsTerm}) = false

function StatsModels.termvars(t::RandomEffectsTerm)
    return vcat(StatsModels.termvars(t.lhs), StatsModels.termvars(t.rhs))
end

StatsModels.terms(t::RandomEffectsTerm) = union(StatsModels.terms(t.lhs), StatsModels.terms(t.rhs))

# | in FakeMixed formula -> RandomEffectsTerm
function StatsModels.apply_schema(t::FunctionTerm{typeof(|)},
                                  schema::Schema,
                                  Mod::Type{<:FakeMixed})
    lhs, rhs = t.args_parsed

    isempty(intersect(StatsModels.termvars(lhs), StatsModels.termvars(rhs))) ||
        throw(ArgumentError("Same variable appears on both sides of |"))

    return apply_schema(RandomEffectsTerm(lhs, rhs), schema, Mod)
end

# allowed types (or tuple thereof) for blocking variables (RHS of |):
const GROUPING_TYPE = Union{<:CategoricalTerm, <:InteractionTerm{<:NTuple{N,CategoricalTerm} where {N}}}
check_re_group_type(term::GROUPING_TYPE) = true
check_re_group_type(terms::Tuple{Vararg{<:GROUPING_TYPE}}) = true
check_re_group_type(x) = false

# make a potentially untyped RandomEffectsTerm concrete
function StatsModels.apply_schema(t::RandomEffectsTerm,
                                  schema::Schema,
                                  Mod::Type{<:FakeMixed})
    lhs, rhs = t.lhs, t.rhs

    # handle intercept in LHS (including checking schema for intercept in another term)
    if (
        !StatsModels.hasintercept(lhs) &&
        !StatsModels.omitsintercept(lhs) &&
        ConstantTerm(1) ∉ schema.already &&
        InterceptTerm{true}() ∉ schema.already
    )
        lhs = InterceptTerm{true}() + lhs
    end

    lhs, rhs = apply_schema.((lhs, rhs), Ref(schema), Mod)

    # check whether grouping terms are categorical or interaction of categorical
    check_re_group_type(rhs) ||
        throw(ArgumentError("blocking variables (those behind |) must be Categorical ($(rhs) is not)"))

    return RandomEffectsTerm(MatrixTerm(lhs), rhs)
end

function StatsModels.modelcols(t::RandomEffectsTerm, d::NamedTuple)
    lhs = t.lhs
    return modelcols(lhs, d) .* -0.0 # make this -0 os that we can identify it
end
