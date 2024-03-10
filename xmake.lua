set_project("ppl++")

add_rules("mode.debug", "mode.release")

set_languages("cxx17")
set_warnings("all")

add_requires("openmp", "glm", "spdlog")

includes("apps")
includes("ppl")

includes("tests")
includes("benchmarks")

-- includes("tmp")
