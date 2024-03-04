add_requires("openmp")

target("ppl")
    set_kind("static")
    add_includedirs("$(projectdir)/include")
    add_files("**/*.cu")
    add_cugencodes("native")
    add_packages("openmp", "glm")
target_end()
