add_requires("gtest")

target("test")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_files("prefix_sum.cu")
    add_cugencodes("native")
    add_packages("gtest")
    add_deps("ppl")
