

target("tmp")
    add_includedirs("$(projectdir)/include")
    add_files("*.cu")
    add_cugencodes("native")
    add_packages("spdlog", "openmp")
    add_deps("ppl")