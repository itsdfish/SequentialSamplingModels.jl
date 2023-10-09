kernel_dist(::Type{Epanechnikov}, w::Float64) = Epanechnikov(0.0, w)
kernel(data) = kde(data; kernel=Epanechnikov)