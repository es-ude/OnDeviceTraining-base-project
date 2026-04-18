
# Host build: native toolchain, minimal HAL shim. Selected example source is
# passed in ${ODT_EXAMPLE_SRC} by the top-level CMakeLists.

add_executable(HOST
    ${ODT_EXAMPLE_SRC}
    ${CMAKE_SOURCE_DIR}/cmake/host/src/hardware_init.c
    ${CMAKE_SOURCE_DIR}/cmake/host/src/debug_lib.c
)
