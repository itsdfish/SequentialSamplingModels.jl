using SafeTestsets

for (root, dirs, files) in walkdir(pwd())
    filter!(f -> occursin(".jl", f) && f ≠ "runtests.jl", files)
    for file in files
        println("file $file")
        path = joinpath(root, file)
        include(path)
    end
end
