target("ppl-hybrid")
    set_kind("static")
    add_includedirs("$(projectdir)/include")
    add_files(
        "**/*.cu", 
        "**.cu", 
        "../openmp/dispatcher.cpp",
        "../openmp/00_init.cpp",
        "../openmp/01_morton.cpp",
        "../openmp/02_sort.cpp",
        "../openmp/04_radix_tree.cpp",
        "../openmp/05_edge_count.cpp",
        "../openmp/07_octree.cpp"
        ) 
    add_cugencodes("native")
    add_packages("openmp", "glm", "spdlog")
