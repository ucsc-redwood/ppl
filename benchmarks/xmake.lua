add_requires("benchmark")

target("benchmark")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_headerfiles("*.cuh", "*.h")
    add_files("*.cu")
    add_packages("benchmark", "glm", "openmp", "spdlog")
    add_cugencodes("native")
    add_deps("pplomp", "ppl-cuda")
target_end()


