# pi0-run: pi0 (PaliGemma + action expert) inference on ggml.
#
# Standalone, against an installed ggml (e.g. qnx-ports/ggml qnx-0.14.0 built with -DGGML_VULKAN=ON):
#   cmake -S packages/pi0-ggml -B build -DCMAKE_PREFIX_PATH=<ggml prefix> && cmake --build build
# Or inside a ggml source tree: copy this directory to ggml/examples/pi0 and add_subdirectory(pi0).
cmake_minimum_required(VERSION 3.14)
project(pi0-ggml CXX)

if (NOT TARGET ggml)
    find_package(ggml REQUIRED)
endif()

add_executable(pi0-run pi0-run.cpp)
target_compile_features(pi0-run PRIVATE cxx_std_17)
if (TARGET ggml::ggml)
    target_link_libraries(pi0-run PRIVATE ggml::ggml)
else()
    target_link_libraries(pi0-run PRIVATE ggml)
endif()

install(TARGETS pi0-run RUNTIME DESTINATION bin)
