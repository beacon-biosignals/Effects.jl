using Documenter
using Effects

makedocs(; modules=[Effects],
         authors="Beacon Biosignals, Inc.",
         repo="https://github.com/beacon-biosignals/Effects.jl/blob/{commit}{path}#{line}",
         sitename="Effects.jl",
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true",
                                canonical="https://beacon-biosignals.github.io/Effects.jl",
                                assets=String[]),
         pages=["Home" => "index.md",
                #   "Partial Effects" => "remef.md",
                "Estimated Marginal Means" => "emmeans.md",
                "API" => "api.md"])

deploydocs(; repo="github.com/beacon-biosignals/Effects.jl",
           devbranch="main",
           push_preview=true)
