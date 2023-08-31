module DynamicPPLExt 
    using SequentialSamplingModels
    using DynamicPPL: reconstruct
    using DynamicPPL: vectorize

    println("loading DynamicPPLExt")
    
    reconstruct(d::SSM2D, v::AbstractVector) = deepcopy(v)
    vectorize(d::SSM2D, r::NamedTuple) = [r...]



end