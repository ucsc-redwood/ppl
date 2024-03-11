add_requires("openmp")

target("ppl")
    set_kind("static")
    add_includedirs("$(projectdir)/include")
    add_files(
        "cuda/*.cu",
        "openmp/*.cpp"
    )
    add_cugencodes("native")
    add_packages("openmp", "glm", "spdlog")
target_end()
