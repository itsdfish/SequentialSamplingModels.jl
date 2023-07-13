module TestModuleExt 
    isdefined(Base, :get_extension) ? (using Plots) : (using ..Plots)
    println("hi")
    function myfunction()

    end

end