target("cpu-only")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_headerfiles("../../include/**/*", "../app_params.hpp")
    add_files(
        "../app_params.cpp",
        "*.cpp")
    add_packages("cli11", "spdlog", "openmp", "glm")
    add_deps("ppl-omp")
