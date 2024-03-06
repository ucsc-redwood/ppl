add_requires("benchmark")

target("benchmark")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_files("benchmark.cu")
    add_packages("benchmark", "glm", "openmp", "spdlog")
    add_cugencodes("native")
    add_deps("ppl")
target_end()


