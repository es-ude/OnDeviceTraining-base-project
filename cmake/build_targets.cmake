
# Detect which platform we are on (pico or stm32)

if(${ODT_TARGET} STREQUAL PICO1)

set(ODT_PICO_PLATFORM 1)

elseif(${ODT_TARGET} STREQUAL PICO2_W)

set(ODT_PICO_PLATFORM 1)
    
elseif(${ODT_TARGET} STREQUAL STM32L4R5XI)

set(ODT_STM32_PLATFORM 1)

elseif(${ODT_TARGET} STREQUAL STM32L476XG)

set(ODT_STM32_PLATFORM 1)

elseif(${ODT_TARGET} STREQUAL STM32F756XG)

set(ODT_STM32_PLATFORM 1)
elseif(${ODT_TARGET} STREQUAL NONE)
message("No ODT target set")
else()

message("No or unkown target defined")

endif()
