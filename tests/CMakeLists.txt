set(CPS_TESTS
    test_fft
    test_fft_extended
    test_windows
    test_filter_design
    test_filter_extended
    test_spectral
    test_generate
    test_peaks
    test_stress
    test_ieee_edge
    test_generate_extended
    test_internals
    test_stubs
    test_adversarial
    test_validation
    test_review_bugs
    test_pocketfft_backend
)

foreach(t ${CPS_TESTS})
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} PRIVATE cppsignal::cppsignal Catch2::Catch2WithMain)
    target_compile_features(${t} PRIVATE cxx_std_20)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# FFTW backend tests — only built when CPS_ENABLE_FFTW=ON
if(CPS_ENABLE_FFTW)
    add_executable(test_fftw_backend test_fftw_backend.cpp)
    target_link_libraries(test_fftw_backend PRIVATE cppsignal::cppsignal Catch2::Catch2WithMain)
    target_compile_features(test_fftw_backend PRIVATE cxx_std_20)
    add_test(NAME test_fftw_backend COMMAND test_fftw_backend)
endif()
