# XXX what about marginalization weights?

# similar to effects (and the underlying math is the same) but
# the establishment of the reference grid is different
# we don't allow specifying the "typifier" here -- if you want that,
# then choose a less full service function
# point people to dof_residual, infinite_dof, and maybe some notes on mixed models
function emmeans(model::RegressionModel; eff_col=nothing, err_col=:err,
                 invlink=identity, levels=Dict(), dof=nothing)
    form = formula(model)
    typical = mean
    # cat_terms = filter(x -> x isa CategoricalTerm, terms(form.rhs))
    # defaults =  Dict(tt.sym => tt.contrasts.levels for tt in cat_terms)
    defaults = Dict{Symbol, Vector}()
    for tt in terms(form.rhs)
        # this kinda feels like a place for dispatch but....
        if tt isa CategoricalTerm
            defaults[tt.sym] = tt.contrasts.levels
        elseif tt isa ContinuousTerm
            defaults[tt.sym] = [tt.mean]
        # handle StandardizdPredictors and other wrapped Terms
        elseif hasproperty(tt, :term) && tt.term isa ContinuousTerm
            defaults[tt.term.sym] = [hasproperty(tt, :center) ? tt.center : tt.term.mean]
        elseif !(tt isa Union{InterceptTerm, InteractionTerm})
            defaults[tt.sym] = [0.0]
        end
    end
    levels = merge(defaults, levels) # prefer user specified levels
    grid = _reference_grid(levels)
    eff_col = string(something(eff_col, _responsename(model)))
    err_col = string(err_col)

    result = effects!(grid, model; eff_col, err_col, typical, invlink)
    if !isnothing(dof)
        result[!, :dof] .= dof(model)
    end
    return result
end

function empairs(model::RegressionModel; eff_col=nothing, err_col=:err,
               invlink=identity, levels=Dict(), dof=nothing, padjust=identity)
    eff_col = something(eff_col, _responsename(model))
    em = emmeans(model; eff_col, err_col, invlink, levels, dof)
    return empairs(em; eff_col, err_col, padjust)
end

pooled_sem(sems...) = sqrt(sum(abs2, sems))
infinite_dof(::RegressionModel) = Inf

# TODO: mark this as experimental and subject to formatting, etc. changes
# TODO point people to https://juliangehring.github.io/MultipleTesting.jl/stable/
function empairs(df::AbstractDataFrame; eff_col, err_col=:err, padjust=identity)
    # need to enforce that we're all the same type
    # (mixing string and symbol is an issue with Not
    #  and a few other things below)
    eff_col = string(eff_col)
    err_col = string(err_col)
    stats_cols = [eff_col, err_col]
    "dof" in names(df) && push!(stats_cols, "dof")

    pairs = combinations(eachrow(df), 2)
    # TODO make this more efficient in allocations
    result_df = mapreduce(vcat, pairs) do (df1, df2)
        result =  Dict{String, Union{String, Number}}()

        for col in names(df1, Not(stats_cols))
            result[col] = if df1[col] == df2[col]
                df1[col]
            else
                string(df1[col], " > ", df2[col])
            end
        end
        result[eff_col] = df1[eff_col] - df2[eff_col]
        result[err_col] = pooled_sem(df1[err_col], df2[err_col])
        if "dof" in names(df)
            df1["dof"] != df2["dof"] &&
                throw(ArgumentError("Incompatible dof found for rows $(df1) and $(df2)"))
            result["dof"] = df1["dof"]
        end
        return DataFrame(result)
    end

    cols = vcat(names(df, Not(stats_cols)), stats_cols)
    select!(result_df, cols)
    if "dof" in stats_cols
        transform!(result_df, [eff_col, err_col] => ByRow(/) => "t")
        transform!(result_df, [:dof, :t] => ByRow() do dof, t
            p = 2 * cdf(TDist(dof), -abs(t))
            return p
        end => "Pr(>|t|)")
        transform!(result_df, "Pr(>|t|)" => padjust => "Pr(>|t|)")
    elseif padjust !== identity
        @warn "padjust specified, but there are no p-values to adjust."
    end
    return result_df
end
