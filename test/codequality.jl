@safetestset "Code Quality" begin

    # check code is formatted
    # @safetestset "code formatting" begin
    #     using JuliaFormatter
    #     using SequentialSamplingModels
    #     @test JuliaFormatter.format(
    #         SequentialSamplingModels; verbose = false, overwrite = false
    #     )
    # end

    # check code quality via Aqua
    @safetestset "Aqua" begin
        using Aqua
        using SequentialSamplingModels
        Aqua.test_all(
            SequentialSamplingModels; ambiguities = false,
            deps_compat = (check_extras = false,)
        )
    end

    # test JET
    @safetestset "JET" begin
        using JET
        using SequentialSamplingModels
        JET.test_package(SequentialSamplingModels; target_defined_modules = true)
    end
end
