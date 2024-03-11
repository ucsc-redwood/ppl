set_project("ppl++")

add_rules("mode.debug", "mode.release")

-- if is_mode("release") then
--     add_defines("NDEBUG")
-- end

set_languages("cxx17")
set_warnings("all")

-- ignore this warning from glm
add_cuflags("-diag-suppress=20012") 

add_requires("openmp", "glm", "spdlog")

includes("apps")
includes("ppl")

-- includes("tests")
includes("benchmarks")

-- includes("tmp")
