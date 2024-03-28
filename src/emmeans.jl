# XXX what about marginalization weights?

_dof(f::Function, model) = f(model)
_dof(x, _) = x

"""
    pooled_sem(sems...)

Compute the pooled standard error of the mean.

This corresponds to the square root of the sum of the
the squared SEMs, which in turn corresponds to the weighted root-mean-square
of the underlying standard deviations.
"""
pooled_sem(sems...) = sqrt(sum(abs2, sems))

# similar to effects (and the underlying math is the same) but
# the establishment of the reference grid is different
# we don't allow specifying the "typifier" here -- if you want that,
# then choose a less full service function
"""
    emmeans(model::RegressionModel; eff_col=nothing, err_col=:err,
            invlink=identity, levels=Dict(), dof=nothing,
            ci_level=nothing, lower_col=:lower, upper_col=:upper)

Compute estimated marginal means, a.k.a. least-square (LS) means for a model.

By default, emmeans are computed for each level of each categorical variable
along with the means of continuous variables. For centered terms, the center
is used instead of the mean. Alternative levels can be specified with `levels`.

`dof` is used to compute the associated degrees of freedom for a particular
margin. For regression models, the appropriate degrees of freedom are
the same degrees of freedom as used for the Wald tests on the coefficients.
These are typically the residual degrees of freedom (e.g., via `dof_residual`).
If `dof` is a function, then it is called on `model` and filled elementwise
into the `dof` column. Alternatively, `dof` can be specified as a single number
or vector of appropriate length. For example, in a mixed effect model with
a large number of observations, `dof=Inf` may be appropriate.

`invlink`, `eff_col` and `err_col` work exactly as in [`effects!`](@ref).

If `ci_level` is provided, then `ci_level` confidence intervals are computed using
the Wald approximation based on the standard errors and quantiles of the ``t``-distribution.
If `dof` is not provided, then the degrees of freedom are assumed to be infinite,
which is equivalent to using the normal distribution.
The corresponding lower and upper edges of the interval are placed in `lower_col`
and `upper_col`, respectively.

Estimated marginal means are closely related to effects and are also known as
least-square means. The functionality here is a convenience
wrapper for [`effects`](@ref) and maps onto the concept of least-square means
as presented in e.g. the [SAS documentation](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.4/statug/statug_glimmix_syntax13.htm).
There are several extensions available to estimated marginal means, related
to when the marginalization occurs and how cells are weighted, but these are
not currently supported. The documentation for the [R package emmeans](https://cran.r-project.org/web/packages/emmeans/index.html)
explains [the background in more depth](https://cran.r-project.org/web/packages/emmeans/vignettes/basics.html).
"""
function emmeans(model::RegressionModel; eff_col=nothing, err_col=:err,
                 invlink=identity, levels=Dict(), dof=nothing, ci_level=nothing,
                 lower_col=:lower, upper_col=:upper)
    form = formula(model)
    typical = mean
    defaults = Dict{Symbol,Vector}()
    for tt in terms(get_matrix_term(form.rhs))
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
    grid = expand_grid(levels)
    eff_col = string(something(eff_col, _responsename(model)))
    err_col = string(err_col)

    result = effects!(grid, model; eff_col, err_col, typical, invlink)
    if !isnothing(dof)
        result[!, :dof] .= _dof(dof, model)
    end
    if !isnothing(ci_level)
        # don't include this in the above if because
        # we don't want to potentially add a dof column if there is no CI
        if isnothing(dof)
            # fall back to z value
            result[!, :dof] .= Inf
        end
        # divide by two for twosided
        scale = abs.(quantile.(TDist.(result[!, :dof]), (1 - ci_level) / 2))
        result[!, lower_col] .= result[!, eff_col] .- result[!, err_col] .* scale
        result[!, upper_col] .= result[!, eff_col] .+ result[!, err_col] .* scale
    end
    return result
end

"""
    empairs(model::RegressionModel; eff_col=nothing, err_col=:err,
            invlink=identity, levels=Dict(), dof=nothing, padjust=identity,
            ci_level=nothing, lower_col=:lower, upper_col=:upper)
    empairs(df::AbstractDataFrame; eff_col, err_col=:err, padjust=identity,
            ci_level=nothing, lower_col=:lower, upper_col=:upper)

Compute pairwise differences of estimated marginal means.

The method for `AbstractDataFrame` acts on the results of [`emmeans`](@ref),
while the method for `RegressionModel` is a convenience wrapper that calls
`emmeans` internally.

The keyword arguments are generally the same as `emmeans`.

By default, pairs are computed for all combinations of categorical variables and
the means/centers of continuous variables. The contrast for a pair "a" vs "b"
is represented in the column for the contrasted variable with `a > b`. For
variables that are the same within a pair (e.g., a continuous variable), the sole
value is displayed as is.

If `dof` is not `nothing`, then p-values are also computed using a t-distribution
with the resulting degrees of freedom. The results for `dof=Inf` correspond to
using a z-distribution, but the column names in the returned dataframe remain
`t` and `Pr(>|t|)`.

If `padjust` is provided, then it is used to compute adjust the p-values for
multiple comparisons. [`MultipleTesting.jl`](https://juliangehring.github.io/MultipleTesting.jl/stable/)
provides a number of useful possibilities for this.

If `ci_level` is provided, then `ci_level` confidence intervals are computed using
the Wald approximation based on the standard errors and quantiles of the ``t``-distribution.
If `dof` is not provided, then the degrees of freedom are assumed to be infinite,
which is equivalent to using the normal distribution.
The corresponding lower and upper edges of the interval are placed in `lower_col`
and `upper_col`, respectively.

!!! note
    `padjust` is silently ignored if `dof` is not provided.

!!! note
    Confidence intervals are **not** adjusted for multiple comparisons.

!!! warning
    This feature is experimental and the precise column names and presentation of
    contrasts/differences may change without being considered breaking.

!!! warning
    The use of `invlink` is subject to a number of interpretation subtleties.
    The EM means are computed on the scale of the linear predictor, then
    transformed to the scale of `invlink`. The associated errors on the
    transformed scale are computed via the difference method. These estimates
    and errors are then used to compute the pairs. Test statistics are
    computed on the scale of these pairs. In general, these will not be the same
    as test statistics computed on the original scale. These subleties are
    discussed in [the documentation for the R package `emmeans`](https://cran.r-project.org/web/packages/emmeans/vignettes/transformations.html).
"""
function empairs(model::RegressionModel; eff_col=nothing, err_col=:err,
                 invlink=identity, levels=Dict(), dof=nothing, padjust=identity,
                 ci_level=nothing, lower_col=:lower, upper_col=:upper)
    eff_col = something(eff_col, _responsename(model))
    em = emmeans(model; eff_col, err_col, invlink, levels, dof)
    return empairs(em; eff_col, err_col, padjust, ci_level, lower_col, upper_col)
end

function empairs(df::AbstractDataFrame; eff_col, err_col=:err, padjust=identity,
                 ci_level=nothing, lower_col=:lower, upper_col=:upper)
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
        result = Dict{String,Union{AbstractString,Number}}()

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
                throw(ArgumentError("Pooled dof not supported: $(df1) and $(df2)"))
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
    end
    if !isnothing(ci_level)
        # don't include this in the above if because
        # we don't want to potentially add a dof column if there is no CI
        if "dof" ∉ stats_cols
            # fall back to z value
            result_df[!, :dof] .= Inf
        end
        # divide by two for twosided
        # 1+level to pull from upper tail
        scale = quantile.(TDist.(result_df[!, :dof]), (1 + ci_level) / 2)
        result_df[!, lower_col] .= result_df[!, eff_col] .- result_df[!, err_col] .* scale
        result_df[!, upper_col] .= result_df[!, eff_col] .+ result_df[!, err_col] .* scale
    end
    return result_df
end
