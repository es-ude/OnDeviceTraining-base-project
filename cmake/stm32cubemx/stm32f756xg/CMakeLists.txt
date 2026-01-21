cmake_minimum_required(VERSION 3.22)

#
# This file is generated only once,
# and is not re-generated if converter is called multiple times.
#
# User is free to modify the file as much as necessary
#

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)


# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set the project name
# set(CMAKE_PROJECT_NAME stm32f756zgtx_playground2)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
# project(${CMAKE_PROJECT_NAME})
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable CMake support for ASM and C languages
# enable_language(C ASM)

# Create an executable object type
#add_executable(${CMAKE_PROJECT_NAME})
add_executable(${ODT_TARGET})

list(APPEND ODT_TARGETS ${ODT_TARGET})
set(ODT_TARGETS "${ODT_TARGETS}" PARENT_SCOPE)


# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# Link directories setup
target_link_directories(${ODT_TARGET} PRIVATE
    # Add user defined library search paths
)

# Add sources to executable
target_sources(${ODT_TARGET} PRIVATE
    # Add user sources here
)

# Add include paths
target_include_directories(${ODT_TARGET} PRIVATE
    # Add user defined include paths
)

# Add project symbols (macros)
target_compile_definitions(${ODT_TARGET} PRIVATE
    # Add user defined symbols
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Add linked libraries
target_link_libraries(${ODT_TARGET}
    stm32cubemx

    # Add user defined libraries
)

# Execute post-build to print size, generate hex and bin
add_custom_command(TARGET ${ODT_TARGET} POST_BUILD
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${ODT_TARGET}>
    COMMAND ${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:${ODT_TARGET}> ${ODT_TARGET}.hex
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${ODT_TARGET}> ${ODT_TARGET}.bin
)