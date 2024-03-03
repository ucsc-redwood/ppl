add_requires("gtest")

for _, file in ipairs(os.files("test_*.cu")) do
    local target_name = path.basename(file)
    target(target_name)
        set_kind("binary")
        add_includedirs("$(projectdir)/include")
        add_files(file)
        add_cugencodes("native")
        add_packages("gtest")
        add_deps("ppl")
end

-- target("test")
--     set_kind("binary")
--     add_includedirs("$(projectdir)/include")
--     add_files("prefix_sum.cu")
--     add_cugencodes("native")
--     add_packages("gtest")
--     add_deps("ppl")

-- target("test2")
--     set_kind("binary")
--     add_includedirs("$(projectdir)/include")
--     add_files("unique.cu")
--     add_cugencodes("native")
--     add_packages("gtest")
--     add_deps("ppl")
