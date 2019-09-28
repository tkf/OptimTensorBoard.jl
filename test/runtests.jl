module TestOptimTensorBoard

using Optim
using OptimTensorBoard
using Test

logdir = joinpath(@__DIR__, "tmp", "logdir")
rm(joinpath(@__DIR__, "tmp"); force=true, recursive=true)

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end
x0 = [0.0, 0.0]
calledwith = []
callback(tr) = (push!(calledwith, tr); false)

@testset "savearrays = true" begin
    empty!(calledwith)
    opt = Optim.Options(
        extended_trace = true,
        callback = optimcallback(
            logdir,
            callback = callback,
            savearrays = true,
        ))
    optimize(f, g!, x0, LBFGS(), opt)
    @test !isempty(calledwith)
end

@testset "savearrays = :image" begin
    empty!(calledwith)
    opt = Optim.Options(
        extended_trace = true,
        callback = optimcallback(
            logdir,
            callback = callback,
            savearrays = :image,
        ))
    optimize(f, g!, x0, LBFGS(), opt)
    @test !isempty(calledwith)
end

end  # module
