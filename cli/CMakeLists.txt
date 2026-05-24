include(FetchContent)

# miniaudio — cross-platform audio I/O, single-header, MIT licensed.
# Its CMakeLists.txt creates a compiled `miniaudio` target that handles
# MINIAUDIO_IMPLEMENTATION and platform link libraries automatically.
FetchContent_Declare(
    miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG        0.11.25
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(miniaudio)

# CLI library — shared logic for generate/filter/play commands, independently testable.
add_library(cps_cli STATIC
    format.cpp
    generate.cpp
    filter.cpp
    play.cpp
)

# Expose CMAKE_SOURCE_DIR so consumers can use #include <cli/format.hpp> etc.
target_include_directories(cps_cli PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(cps_cli PUBLIC cppsignal::cppsignal)
target_link_libraries(cps_cli PRIVATE miniaudio)
target_compile_features(cps_cli PRIVATE cxx_std_20)

# cps binary
add_executable(cps main.cpp)
target_link_libraries(cps PRIVATE cps_cli)
target_compile_features(cps PRIVATE cxx_std_20)
