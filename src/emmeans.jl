# XXX what about df?
# XXX what about test statistics?
# XXX what about MC-corrected p-values?

# similar to effects (and the underlying math is the same) but
# the establishment of the reference grid is different
# we don't allow specifying the "typifier" here -- if you want that,
# then choose a less full service function
function emmeans(model::RegressionModel; eff_col=nothing, err_col=:err,
                 invlink=identity, levels=Dict())
    form = formula(model)
    # cat_terms = filter(x -> x isa CategoricalTerm, terms(form.rhs))
    # defaults =  Dict(tt.sym => tt.contrasts.levels for tt in cat_terms)
    defaults = Dict{Symbol, Vector}()
    for tt in terms(form.rhs)
        # this kinda feels like a place for dispatch but....
        if tt isa CategoricalTerm
            defaults[tt.sym] = tt.contrasts.levels
        elseif tt isa ContinuousTerm
            defaults[tt.sym] = [tt.mean]
        end
    end
    levels = merge(defaults, levels) # prefer user specified levels
    grid = _reference_grid(levels)
    dv = something(eff_col, _responsename(model))
    # XXX what about df?
    return effects!(grid, model; eff_col=dv, err_col, typical=mean, invlink)
end

function empairs(model::RegressionModel; eff_col=nothing, err_col=:err,
               invlink=identity, levels=Dict())
    eff_col = something(eff_col, _responsename(model))
    em = emmeans(model; eff_col, err_col, invlink, levels)
    return empairs(em; eff_col, err_col,)
end

pooled_sem(x...) = sqrt(sum(abs2, x))

# TODO: mark this as experimental and subject to formatting, etc. changes
function empairs(df::AbstractDataFrame; eff_col, err_col=:err)
    # need to enforce that we're all the same type
    # (mixing string and symbol is an issue with Not
    #  and a few other things below)
    eff_col = string(eff_col)
    err_col = string(err_col)

    pairs = combinations(eachrow(df), 2)
    # TODO make this more efficient in allocations
    result_df = mapreduce(vcat, pairs) do (df1, df2)
        result =  Dict{String, Union{String, Number}}()

        for col in names(df1, Not([eff_col, err_col]))
            result[col] = if df1[col] == df2[col]
                df1[col]
            else
                string(df1[col], ">", df2[col])
            end
        end
        result[eff_col] = df1[eff_col] - df2[eff_col]
        result[err_col] = pooled_sem(df1[err_col], df2[err_col])
        return DataFrame(result)
    end
    return select!(result_df, names(df, Not([eff_col, err_col])), eff_col, err_col)
end
