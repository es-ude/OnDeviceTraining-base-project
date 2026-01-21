cmake_minimum_required(VERSION 3.16...4.1)

include(FetchContent)
include(cmake/build_targets.cmake)


if(${ODT_PICO_PLATFORM})
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pico/pico_pre.cmake)
endif()

project(BA C ASM CXX)

if(${ODT_PREPARE})
    FetchContent_Declare(
            pico_sdk
            GIT_REPOSITORY https://github.com/raspberrypi/pico-sdk
            GIT_TAG 2.2.0
    )
    FetchContent_Declare(
            odt
            GIT_REPOSITORY https://github.com/es-ude/OnDeviceTraining
            GIT_TAG main
    )


    FetchContent_Populate(
            pico_sdk
            QUIET
            GIT_REPOSITORY https://github.com/raspberrypi/pico-sdk
            GIT_TAG 2.2.0
            GIT_SUBMODULES_RECURSE TRUE

            SOURCE_DIR ${CMAKE_SOURCE_DIR}/pico-sdk/src
            BINARY_DIR ${FETCHCONTENT_BASE_DIR}/pico-sdk/build
            SUBBUILD_DIR ${FETCHCONTENT_BASE_DIR}/pico-sdk/subbuild
                
    )
    FetchContent_Populate(
            odt
            QUIET
            GIT_REPOSITORY https://github.com/es-ude/OnDeviceTraining
            GIT_TAG "main"
            GIT_SUBMODULES_RECURSE FALSE

            SOURCE_DIR ${CMAKE_SOURCE_DIR}/OnDeviceTraining/src
            BINARY_DIR ${CMAKE_SOURCE_DIR}/OnDeviceTraining/build
            SUBBUILD_DIR ${CMAKE_SOURCE_DIR}/OnDeviceTraining/subbuild
           )
else()

# Each target has to include these libs
set(ODT_LIBS
            Layer
            Tensor
            Rounding
            TensorAPI
            InferenceAPI
            ReluAPI
            QuantizationAPI
            TensorConversion
            LossFunction
            LinearAPI
            Linear
            Sgd
            SgdAPI
            StorageAPI
            TrainingAPI
            Optimizer
            OptimizerAPI
            MSE
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/OnDeviceTraining/src)

if(${ODT_PICO_PLATFORM})
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pico/pico_post.cmake)
elseif(${ODT_TARGET} STREQUAL "STM32F756XG")
add_subdirectory(cmake/stm32cubemx/stm32f756xg)
elseif(${ODT_TARGET} STREQUAL "STM32L476XG")
add_subdirectory(cmake/stm32cubemx/stm32l476xg)
elseif(${ODT_TARGET} STREQUAL "STM32L4R5XI")
add_subdirectory(cmake/stm32cubemx/stm32l4r5zi)
endif()

endif()