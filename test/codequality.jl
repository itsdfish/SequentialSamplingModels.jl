@safetestset "Code Quality" begin

    # check code quality via Aqua
    @safetestset "Aqua" begin
        using Aqua
        using SequentialSamplingModels
        Aqua.test_all(
            SequentialSamplingModels;
            ambiguities = false,
            deps_compat = (check_extras = false,),
            project_extras = false
        )
    end

    # test JET
    @safetestset "JET" begin
        using JET
        using SequentialSamplingModels
        JET.test_package(
            SequentialSamplingModels;
            target_modules = (SequentialSamplingModels,)
        )
    end
end
