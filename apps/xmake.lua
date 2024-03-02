add_requires("cli11", "spdlog", "openmp", "glm")

target("app")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_files(
        "main.cu",
        "gpu_kernels/*.cu"
        )
    add_packages("cli11", "spdlog", "openmp", "glm")
    add_cugencodes("native")
    -- add_deps("ppl")
    -- for host compiler, set openmp
    add_cxxflags("-fopenmp")
    add_cuflags("-Xcompiler -fopenmp")
