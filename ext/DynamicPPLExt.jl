module DynamicPPLExt

    using SequentialSamplingModels
    using DynamicPPL
    import DynamicPPL: vectorize
    import DynamicPPL: reconstruct

    vectorize(d::SSM2D, r::NamedTuple) = [r...]
    reconstruct(d::SSM2D, val::NamedTuple) = deepcopy(val)

end