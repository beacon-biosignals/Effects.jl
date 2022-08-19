module Effects

using Base.Iterators: product
using Combinatorics: combinations
using DataFrames
using Distributions
using LinearAlgebra
using StatsAPI
using Statistics
using StatsModels
using StatsBase
using Tables

using StatsModels: AbstractTerm
using ForwardDiff

include("typical.jl")
include("regressionmodel.jl")
export effects, effects!

include("emmeans.jl")
export emmeans, empairs, infinite_dof

end # module
