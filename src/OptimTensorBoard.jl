module OptimTensorBoard

# Use README as the docstring of the module:
@doc read(joinpath(dirname(@__DIR__), "README.md"), String) OptimTensorBoard

export optimcallback

using Logging
using TensorBoardLogger

"""
    optimcallback(logger; [callback, savearrays])
    optimcallback(tblogger_args...; [callback, savearrays], [tblogger_kwargs...])

Create a callable object that is suitable for passing it to `Optim.Options`.
`callback` is a callable that is called after this callback is called.
"""
optimcallback(logger::AbstractLogger; callback=returnfalse, savearrays=true) =
    OptimCallback(logger, callback, savearrays)

optimcallback(args...; callback=returnfalse, savearrays=true, kwargs...) =
    OptimCallback(TBLogger(args...; kwargs...), callback, savearrays)

returnfalse(_) = false

@enum ArrayHandler DONTSAVE HISTOGRAM IMAGE

# Is it a good idea?
Base.convert(::Type{ArrayHandler}, yes::Bool) = yes ? HISTOGRAM : DONTSAVE
function Base.convert(::Type{ArrayHandler}, name::Symbol)
    name == :histogram && return HISTOGRAM
    name == :image && return IMAGE
    throw(ArgumentError("Unknown array handler: $name"))
end

struct OptimCallback
    logger::AbstractLogger
    callback
    savearrays::ArrayHandler
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
            elseif cb.savearrays === HISTOGRAM && value isa AbstractArray{<:Real}
                log_histogram(logger, "optim/metadata/$key", vec(collect(value)))
            elseif (
                cb.savearrays === IMAGE &&
                value isa AbstractArray{<:Real} &&
                ndims(value) in (1, 2, 3)
            )
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
