cmake_minimum_required(VERSION 3.20)
project(cppsignal VERSION 0.1.0 LANGUAGES CXX)

# ─────────────────────────────────────────────────────────────────────────────
# CppSignal — a modern C++20 signal processing library
#
# Design goals:
#   • scipy.signal-style function API: butter(4, 100, Lowpass, {.fs=1000})
#   • Pluggable backends for FFT, threading, and memory allocation
#   • Header-only: nothing to compile, just FetchContent + link
#   • MIT licensed: no copyleft, usable in commercial code
#
# Quick start (FetchContent):
#   FetchContent_Declare(cppsignal GIT_REPOSITORY ... GIT_TAG main)
#   FetchContent_MakeAvailable(cppsignal)
#   target_link_libraries(myapp PRIVATE cppsignal::cppsignal)
# ─────────────────────────────────────────────────────────────────────────────

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC debug-info format fix ─────────────────────────────────────────────────
# The default /Zi flag tells cl.exe to write all debug info into a single
# shared .pdb file per project.  When MSBuild compiles many .cpp files in
# parallel they all race to write that one file → C1041.
#
# /Z7 embeds debug symbols directly in each .obj file instead.  There is no
# shared PDB during compilation, so the race is impossible.  Debug symbols are
# still fully available; the linker collects them from the .obj files.
#
# We patch the CMake cache variables with FORCE so every target — including
# those created inside FetchContent subprojects (Catch2, pocketfft, …) —
# picks up /Z7 before their vcxproj files are generated.
if(MSVC)
    foreach(_lang CXX C)
        foreach(_cfg DEBUG RELWITHDEBINFO)
            set(_var "CMAKE_${_lang}_FLAGS_${_cfg}")
            string(REPLACE "/Zi" "/Z7" _patched "${${_var}}")
            string(REPLACE "/ZI" "/Z7" _patched "${_patched}")
            if(NOT _patched MATCHES "/Z7")
                string(APPEND _patched " /Z7")
            endif()
            set(${_var} "${_patched}" CACHE STRING "" FORCE)
        endforeach()
    endforeach()
    unset(_lang)
    unset(_cfg)
    unset(_var)
    unset(_patched)
endif()

# ── Main interface target ────────────────────────────────────────────────────
# INTERFACE = header-only: no compiled .cpp files, only include paths and flags
add_library(cppsignal INTERFACE)
add_library(cppsignal::cppsignal ALIAS cppsignal)

target_include_directories(cppsignal INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Require C++20 from any target that links us
target_compile_features(cppsignal INTERFACE cxx_std_20)

# ── PocketFFT backend (default, always available) ────────────────────────────
# PocketFFT is a single-header MIT-licensed FFT library by Martin Reinecke.
# Used internally by NumPy since 2019. We fetch it automatically via CMake.
include(FetchContent)

FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
# FetchContent_MakeAvailable populates the source and (if pocketfft has a
# CMakeLists.txt) calls add_subdirectory. We only need the header, so we
# override SOURCE_SUBDIR to point at a non-existent path, which silently
# skips add_subdirectory while still giving us pocketfft_SOURCE_DIR.
# The $<BUILD_INTERFACE:...> wrapper tells CMake this path is only valid
# during local builds, satisfying the INTERFACE target path restriction.
FetchContent_MakeAvailable(pocketfft)
# pocketfft has no CMakeLists.txt — MakeAvailable just populates the source
# directory. We add it to our include path so pocketfft_hdronly.h is found.
target_include_directories(cppsignal INTERFACE
    $<BUILD_INTERFACE:${pocketfft_SOURCE_DIR}>
)

# ── Optional FFTW3 backend ───────────────────────────────────────────────────
# FFTW is GPL-licensed. Enable only if you accept that licence or hold a
# commercial FFTW licence. Off by default so the library stays MIT-clean.
option(CPS_ENABLE_FFTW "Enable FFTW3 backend (GPL — read licence before enabling)" OFF)
if(CPS_ENABLE_FFTW)
    find_package(FFTW3 REQUIRED)
    target_link_libraries(cppsignal INTERFACE FFTW3::fftw3)
    target_compile_definitions(cppsignal INTERFACE CPS_HAS_FFTW)
    message(STATUS "cppsignal: FFTW3 backend enabled (GPL licence applies)")
endif()

# ── Optional C++23 std::execution threading backend ─────────────────────────
option(CPS_ENABLE_STD_EXECUTION "Enable C++23 std::execution threading backend" OFF)
if(CPS_ENABLE_STD_EXECUTION)
    target_compile_definitions(cppsignal INTERFACE CPS_HAS_STD_EXECUTION)
endif()

# ── Code coverage ────────────────────────────────────────────────────────────
# Adds a 'coverage' build target that runs the full test suite and produces an
# HTML report + Cobertura XML in <build>/coverage_report/.
#
# Windows / MSVC  →  requires OpenCppCoverage (runtime instrumentation, no
#                    recompile flags needed):
#                    choco install opencppcoverage
#                    https://github.com/OpenCppCoverage/OpenCppCoverage/releases
#
# GCC / Clang     →  compiles tests with --coverage and requires gcovr:
#                    pip install gcovr   or   apt install gcovr
#
# Usage:
#   cmake -S cppsignal -B cppsignal-build -DCPS_ENABLE_COVERAGE=ON
#   cmake --build cppsignal-build --config Debug --target coverage
option(CPS_ENABLE_COVERAGE "Enable coverage instrumentation and add 'coverage' target" OFF)

if(CPS_ENABLE_COVERAGE AND NOT MSVC)
    # GCC/Clang: --coverage flags must reach the test executable targets, so
    # set them before add_subdirectory(tests) is called below.
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
option(CPS_BUILD_TESTS "Build cppsignal test suite" ON)
if(CPS_BUILD_TESTS)
    enable_testing()

    # Catch2 v3 — modern header-only test framework
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.3
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)

    add_subdirectory(tests)
    message(STATUS "cppsignal: tests enabled")
endif()

# ── Coverage target (defined after tests so targets already exist) ────────────
if(CPS_ENABLE_COVERAGE)
    if(NOT CPS_BUILD_TESTS)
        message(FATAL_ERROR "CPS_ENABLE_COVERAGE=ON requires CPS_BUILD_TESTS=ON")
    endif()

    if(MSVC)
        # ── Windows / MSVC: OpenCppCoverage ───────────────────────────────────
        find_program(OPENCPPCOVERAGE_EXE OpenCppCoverage
            HINTS
                "C:/Program Files/OpenCppCoverage"
                "$ENV{ProgramFiles}/OpenCppCoverage"
                "$ENV{LOCALAPPDATA}/Programs/OpenCppCoverage"
        )
        if(NOT OPENCPPCOVERAGE_EXE)
            message(FATAL_ERROR
                "CPS_ENABLE_COVERAGE=ON requires OpenCppCoverage on Windows/MSVC.\n"
                "  choco install opencppcoverage\n"
                "  — or — download from https://github.com/OpenCppCoverage/OpenCppCoverage/releases"
            )
        endif()

        # OpenCppCoverage requires native Windows backslash paths.
        # Use file(TO_NATIVE_PATH) to convert CMAKE_SOURCE_DIR (which uses /).
        # Use relative output paths so "html:coverage_report" avoids any ambiguity
        # with the drive-letter colon in absolute paths like "html:C:\...".
        file(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}/include" _occ_sources)

        add_custom_target(coverage
            # Build all test executables first, then run coverage
            COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
            COMMAND "${OPENCPPCOVERAGE_EXE}"
                --sources "${_occ_sources}"
                --export_type "html:coverage_report"
                --export_type "cobertura:coverage.xml"
                --cover_children
                --
                ctest
                    "--test-dir" "${CMAKE_BINARY_DIR}"
                    "-C" "$<CONFIG>"
                    "--output-on-failure"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            COMMENT "Coverage → ${CMAKE_BINARY_DIR}/coverage_report/index.html"
            VERBATIM
        )
        message(STATUS "cppsignal: 'coverage' target enabled (OpenCppCoverage @ ${OPENCPPCOVERAGE_EXE})")

    else()
        # ── GCC / Clang: gcov + gcovr ─────────────────────────────────────────
        find_program(GCOVR_EXE gcovr)
        if(NOT GCOVR_EXE)
            message(FATAL_ERROR
                "CPS_ENABLE_COVERAGE=ON requires gcovr.\n"
                "  pip install gcovr   or   sudo apt install gcovr"
            )
        endif()

        add_custom_target(coverage
            # Run the tests (executables were built with --coverage)
            COMMAND ctest "--test-dir" "${CMAKE_BINARY_DIR}" "--output-on-failure"
            # Harvest .gcda files and produce the report
            COMMAND "${GCOVR_EXE}"
                --root "${CMAKE_SOURCE_DIR}"
                --filter "${CMAKE_SOURCE_DIR}/include"
                --html-details "${CMAKE_BINARY_DIR}/coverage_report/index.html"
                --xml "${CMAKE_BINARY_DIR}/coverage.xml"
                --print-summary
                "${CMAKE_BINARY_DIR}"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            COMMENT "Coverage → ${CMAKE_BINARY_DIR}/coverage_report/index.html"
            VERBATIM
        )
        message(STATUS "cppsignal: 'coverage' target enabled (gcovr @ ${GCOVR_EXE})")
    endif()
endif()

# ── Examples ─────────────────────────────────────────────────────────────────
option(CPS_BUILD_EXAMPLES "Build cppsignal examples" OFF)
if(CPS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ── Install rules (for system-wide install or vcpkg/conan packaging) ─────────
include(GNUInstallDirs)
install(TARGETS cppsignal EXPORT cppsignalTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT cppsignalTargets
    FILE cppsignalTargets.cmake
    NAMESPACE cppsignal::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cppsignal
)
