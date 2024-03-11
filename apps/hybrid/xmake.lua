target("hybrid")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    -- add_headerfiles("../include/**/*", "../app_params.hpp")
    add_files(
        "*.cu",
        "../app_params.cpp")
    add_packages("cli11", "spdlog", "openmp", "glm")
    add_cugencodes("native")
    add_deps("ppl-hybrid")
    -- for host compiler, set openmp
    add_cxxflags("-fopenmp")
    add_cuflags("-Xcompiler -fopenmp")
