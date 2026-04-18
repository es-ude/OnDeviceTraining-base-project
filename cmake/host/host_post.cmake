
# Host build wiring. Adds the core ODT libs plus the host-only extras
# (NPYLoader + CSVHelper) which depend on the filesystem and are not
# linked into MCU builds.

target_link_libraries(HOST PRIVATE
    ${ODT_LIBS}
    ${ODT_LIBS_HOST_EXTRA}
    m
)

target_compile_definitions(HOST PRIVATE
    DEBUG_MODE_ERROR
)

target_include_directories(HOST PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/examples
)
