add_executable(lowpass_filter  lowpass_filter.cpp)
add_executable(spectrogram_ex  spectrogram.cpp)

foreach(ex lowpass_filter spectrogram_ex)
    target_link_libraries(${ex} PRIVATE cppsignal::cppsignal)
    target_compile_features(${ex} PRIVATE cxx_std_20)
endforeach()
