target("bench-gpu")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_headerfiles("cu_bench_helper.cuh", "../config.h")
    add_files("benchmark.cu")
    add_packages("benchmark", "glm", "openmp", "spdlog")
    add_cugencodes("native")
    add_deps("ppl-hybrid")

