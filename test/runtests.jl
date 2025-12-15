using SafeTestsets

for (root, dirs, files) in walkdir(pwd())
    filter!(f -> occursin(".jl", f) && f ≠ "runtests.jl", files)
    for file in files
        path = joinpath(root, file)
        include(path)
    end
end
