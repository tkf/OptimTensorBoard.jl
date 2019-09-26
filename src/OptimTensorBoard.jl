module OptimTensorBoard

# Use README as the docstring of the module:
@doc read(joinpath(dirname(@__DIR__), "README.md"), String) OptimTensorBoard

export optimcallback

using Logging
using TensorBoardLogger

"""
    optimcallback(logger; [callback])
    optimcallback(tblogger_args...; [callback], [tblogger_kwargs...])

Create a callable object that is suitable for passing it to `Optim.Options`.
`callback` is a callable that is called after this callback is called.
"""
optimcallback(logger::AbstractLogger; callback=returnfalse) =
    OptimCallback(logger, callback)

optimcallback(args...; callback=returnfalse, kwargs...) =
    OptimCallback(TBLogger(args...; kwargs...), callback)

returnfalse(_) = false

struct OptimCallback
    logger::AbstractLogger
    callback
end

function (cb::OptimCallback)(tr)
    logger = cb.logger

    if tr isa AbstractArray
        os = tr[end]
    else
        os = tr
    end
    # os :: OptimizationState

    with_logger(logger) do
        @info "optim" iteration=os.iteration value=os.value g_norm=os.g_norm
    end
    if logger isa TBLogger
        for (key, value) in os.metadata
            if value isa Union{Real, Complex}
                log_value(logger, "optim/metadata/$key", value)
            elseif value isa AbstractArray && ndims(value) in (1, 2, 3)
                fmt = (L, HW, HWC)[ndims(value)]
                minval, maxval = extrema(value)
                if minval != maxval
                    image = (value .- minval) ./ (maxval - minval)
                    log_image(logger, "optim/metadata/image/$key", image, fmt)
                end
                log_value(logger, "optim/metadata/min/$key", minval)
                log_value(logger, "optim/metadata/max/$key", maxval)
            end
        end
    end

    return cb.callback(tr)
end
# For definition of `OptimizationState`, see:
# https://github.com/JuliaNLSolvers/Optim.jl/blob/v0.19.3/src/types.jl#L155-L160

end # module
