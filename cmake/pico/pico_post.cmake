pico_sdk_init()

if(${ODT_TARGET} STREQUAL "PICO1")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pico/targets/pico1.cmake)
elseif(${ODT_TARGET} STREQUAL "PICO2_W")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pico/targets/pico2w.cmake)
endif()