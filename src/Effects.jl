module Effects

using Base.Iterators: product
using DataFrames
using LinearAlgebra
using Statistics
using StatsModels
using StatsBase
using Tables

using StatsModels: AbstractTerm
using ForwardDiff

export effects
export effects!

include("typical.jl")
include("regressionmodel.jl")

end # module
