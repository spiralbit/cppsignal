# CLI library — shared logic for generate/filter commands, independently testable.
add_library(cps_cli STATIC
    format.cpp
    generate.cpp
    filter.cpp
)

# Expose CMAKE_SOURCE_DIR so consumers can use #include <cli/format.hpp> etc.
target_include_directories(cps_cli PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(cps_cli PUBLIC cppsignal::cppsignal)
target_compile_features(cps_cli PRIVATE cxx_std_20)

# cps binary
add_executable(cps main.cpp)
target_link_libraries(cps PRIVATE cps_cli)
target_compile_features(cps PRIVATE cxx_std_20)
