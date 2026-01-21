#include "debug_lib.h"
#include "gpio.h"

void debug_sleep(int ms) {
    HAL_Delay(ms);    
}

void debug_toggle_user_led() {
    HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_7);
    HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_14);
}