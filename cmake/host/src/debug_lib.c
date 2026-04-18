#include "debug_lib.h"

#include <unistd.h>

void debug_sleep(int ms) {
    if (ms > 0) usleep((useconds_t)ms * 1000U);
}

void debug_toggle_user_led(void) {
}
