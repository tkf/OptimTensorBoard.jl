using Documenter, OptimTensorBoard

makedocs(;
    modules=[OptimTensorBoard],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/tkf/OptimTensorBoard.jl/blob/{commit}{path}#L{line}",
    sitename="OptimTensorBoard.jl",
    authors="Takafumi Arakaki <aka.tkf@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/tkf/OptimTensorBoard.jl",
)
