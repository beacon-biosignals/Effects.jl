module Effects

using Base.Iterators: product
using DataFrames
using LinearAlgebra
using Statistics
using StatsModels
using StatsBase
using Tables

export effects
export effects!

include("regressionmodel.jl")

end # module
