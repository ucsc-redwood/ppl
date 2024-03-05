add_requires("gtest", "spdlog")

for _, file in ipairs(os.files("test_*.cu")) do
    local target_name = path.basename(file)
    target(target_name)
        set_kind("binary")
        add_includedirs("$(projectdir)/include")
        add_files(file)
        add_cugencodes("native")
        add_packages("gtest", "spdlog", "glm")
        add_deps("ppl")
end
