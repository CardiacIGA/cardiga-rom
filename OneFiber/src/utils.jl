# This makes it so that `AxisArrays.jl` now respects the ordering of
# the symbols when indexing, e.g. `A[[:a, :b]]` and `A[[:b, :a]]` will
# return the reversed ordering + now stuff like `A[[:a, :a]]` will also
# have the expected behavior.
function AxisArrays.axisindexes(::Type{AxisArrays.Categorical}, ax::AbstractVector, idx::AbstractVector)
    # res = findall(in(idx), ax) # <= original impl
    res = mapreduce(vcat, idx) do i
        findfirst(isequal(i), ax)
    end
    length(res) == length(idx) || throw(ArgumentError("index $(setdiff(idx,ax)) not found"))
    res
end

# Essentially just collapses the `syms` and then reshapes.
# The ordering of `syms` is now preserved.
function Base.Array(
    chains::MCMCChains.Chains,
    syms::AbstractArray{Symbol},
    args...;
    kwargs...
)
    # HACK> Index into `AxisArray` directly rather than chain because
    # chain will not respect ordering of indices like `AxisArray` does.
    a = Array(chains.value[:, vec(syms), :], args...; kwargs...)
    return reshape(a, size(a, 1), size(syms)..., size(a, 3))
end
