
set(PICO_SDK_FETCH_FROM_GIT off)
# set(PICO_SDK_FETCH_FROM_GIT_TAG ${tag})
set(PICO_SDK_FETCH_FROM_GIT_PATH ${CMAKE_SOURCE_DIR}/pico_sdk)
# include(cmake/pico/pico_sdk_import.cmake)
# include(${CMAKE_SOURCE_DIR}/pico-sdk/src/pico_sdk_init.cmake)

if(${ODT_TARGET} STREQUAL "PICO1")
    set(PICO_BOARD pico)

    add_executable(PICO1
        ${ODT_EXAMPLE_SRC}
        ${CMAKE_SOURCE_DIR}/cmake/pico/src/hardware_init.c
        ${CMAKE_SOURCE_DIR}/cmake/pico/src/debug_lib.c
    )

elseif(${ODT_TARGET} STREQUAL "PICO2_W")
    set(PICO_BOARD pico2_w)

    add_executable(PICO2_W
        ${ODT_EXAMPLE_SRC}
        ${CMAKE_SOURCE_DIR}/cmake/pico/src/hardware_init.c
        ${CMAKE_SOURCE_DIR}/cmake/pico/src/debug_lib.c
    )

endif()

include(${CMAKE_SOURCE_DIR}/pico-sdk/src/pico_sdk_init.cmake)
