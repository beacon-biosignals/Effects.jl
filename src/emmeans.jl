# XXX what about marginalization weights?

_dof(f::Function, model) = f(model)
_dof(x, _) = x

pooled_sem(sems...) = sqrt(sum(abs2, sems))

# similar to effects (and the underlying math is the same) but
# the establishment of the reference grid is different
# we don't allow specifying the "typifier" here -- if you want that,
# then choose a less full service function
"""
    emmeans(model::RegressionModel; eff_col=nothing, err_col=:err,
                    invlink=identity, levels=Dict(), dof=nothing)

Compute estimated martginal means, a.k.a. least-squate (LS) means for a model.

By default, emmeans are computed for each level of each categorical variable
along with the means of continuous variables. For centered terms, the center
is used instead of the mean. Alternative levels can be specified with `levels`.

Estimated marginal means are closely related to effects and can be viewed as a
generalization of least-square means. The functionality here is a convenience
wrapper for [`effects`](@ref) and maps onto the concept of least-square means
as presented in e.g. the [SAS documentation](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.4/statug/statug_glimmix_syntax13.htm).
There are several extensions available to estimated marginal means, related
to when the marginalization occurs and how cells are weighted, but these are
not currently supported. The documentation for the [R package emmeans](https://cran.r-project.org/web/packages/emmeans/index.html)
explains [the background in more depth](https://cran.r-project.org/web/packages/emmeans/vignettes/basics.html).


"""# point people to dof_residual, infinite_dof, and maybe some notes on mixed models
function emmeans(model::RegressionModel; eff_col=nothing, err_col=:err,
                 invlink=identity, levels=Dict(), dof=nothing)
    form = formula(model)
    typical = mean
    # cat_terms = filter(x -> x isa CategoricalTerm, terms(form.rhs))
    # defaults =  Dict(tt.sym => tt.contrasts.levels for tt in cat_terms)
    defaults = Dict{Symbol,Vector}()
    for tt in terms(form.rhs)
        # this kinda feels like a place for dispatch but....
        if tt isa CategoricalTerm
            defaults[tt.sym] = tt.contrasts.levels
        elseif tt isa ContinuousTerm
            defaults[tt.sym] = [tt.mean]
            # handle StandardizdPredictors and other wrapped Terms
        elseif hasproperty(tt, :term) && tt.term isa ContinuousTerm
            defaults[tt.term.sym] = [hasproperty(tt, :center) ? tt.center : tt.term.mean]
        end
    end
    levels = merge(defaults, levels) # prefer user specified levels
    grid = _reference_grid(levels)
    eff_col = string(something(eff_col, _responsename(model)))
    err_col = string(err_col)

    result = effects!(grid, model; eff_col, err_col, typical, invlink)
    if !isnothing(dof)
        result[!, :dof] .= _dof(dof, model)
    end
    return result
end

function empairs(model::RegressionModel; eff_col=nothing, err_col=:err,
                 invlink=identity, levels=Dict(), dof=nothing, padjust=identity)
    eff_col = something(eff_col, _responsename(model))
    em = emmeans(model; eff_col, err_col, invlink, levels, dof)
    return empairs(em; eff_col, err_col, padjust)
end

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
        result = Dict{String,Union{String,Number}}()

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
        transform!(result_df,
                   [:dof, :t] => ByRow() do dof, t
                       p = 2 * cdf(TDist(dof), -abs(t))
                       return p
                   end => "Pr(>|t|)")
        transform!(result_df, "Pr(>|t|)" => padjust => "Pr(>|t|)")
    elseif padjust !== identity
        @warn "padjust specified, but there are no p-values to adjust."
    end
    return result_df
end
