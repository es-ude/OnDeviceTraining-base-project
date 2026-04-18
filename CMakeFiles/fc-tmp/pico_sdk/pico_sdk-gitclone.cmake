# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

if(EXISTS "/Users/leo/work/OnDeviceTraining-base-project/CMakeFiles/fc-stamp/pico_sdk/pico_sdk-gitclone-lastrun.txt" AND EXISTS "/Users/leo/work/OnDeviceTraining-base-project/CMakeFiles/fc-stamp/pico_sdk/pico_sdk-gitinfo.txt" AND
  "/Users/leo/work/OnDeviceTraining-base-project/CMakeFiles/fc-stamp/pico_sdk/pico_sdk-gitclone-lastrun.txt" IS_NEWER_THAN "/Users/leo/work/OnDeviceTraining-base-project/CMakeFiles/fc-stamp/pico_sdk/pico_sdk-gitinfo.txt")
  message(VERBOSE
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/Users/leo/work/OnDeviceTraining-base-project/CMakeFiles/fc-stamp/pico_sdk/pico_sdk-gitclone-lastrun.txt'"
  )
  return()
endif()

# Even at VERBOSE level, we don't want to see the commands executed, but
# enabling them to be shown for DEBUG may be useful to help diagnose problems.
cmake_language(GET_MESSAGE_LOG_LEVEL active_log_level)
if(active_log_level MATCHES "DEBUG|TRACE")
  set(maybe_show_command COMMAND_ECHO STDOUT)
else()
  set(maybe_show_command "")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/Users/leo/work/OnDeviceTraining-base-project/pico-sdk/src"
  RESULT_VARIABLE error_code
  ${maybe_show_command}
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/Users/leo/work/OnDeviceTraining-base-project/pico-sdk/src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/nix/store/6qbj40r0s289k5slmy8yna5x2hfz01wg-git-2.53.0/bin/git"
            clone --no-checkout --config "advice.detachedHead=false" "https://github.com/raspberrypi/pico-sdk" "src"
    WORKING_DIRECTORY "/Users/leo/work/OnDeviceTraining-base-project/pico-sdk"
    RESULT_VARIABLE error_code
    ${maybe_show_command}
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(NOTICE "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/raspberrypi/pico-sdk'")
endif()

execute_process(
  COMMAND "/nix/store/6qbj40r0s289k5slmy8yna5x2hfz01wg-git-2.53.0/bin/git"
          checkout "2.2.0" --
  WORKING_DIRECTORY "/Users/leo/work/OnDeviceTraining-base-project/pico-sdk/src"
  RESULT_VARIABLE error_code
  ${maybe_show_command}
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '2.2.0'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/nix/store/6qbj40r0s289k5slmy8yna5x2hfz01wg-git-2.53.0/bin/git" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "/Users/leo/work/OnDeviceTraining-base-project/pico-sdk/src"
    RESULT_VARIABLE error_code
    ${maybe_show_command}
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/Users/leo/work/OnDeviceTraining-base-project/pico-sdk/src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/Users/leo/work/OnDeviceTraining-base-project/CMakeFiles/fc-stamp/pico_sdk/pico_sdk-gitinfo.txt" "/Users/leo/work/OnDeviceTraining-base-project/CMakeFiles/fc-stamp/pico_sdk/pico_sdk-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  ${maybe_show_command}
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/Users/leo/work/OnDeviceTraining-base-project/CMakeFiles/fc-stamp/pico_sdk/pico_sdk-gitclone-lastrun.txt'")
endif()
