cmake_minimum_required(VERSION 3.15)
project(EdgeAI_ECG_Firmware C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optimization and warning flags
add_compile_options(-Wall -Wextra -O2)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

# Source files
set(SOURCES
    src/main.c
    src/ecg_dsp.c
    src/state_scheduler.c
    src/secure_telemetry.c
)

# Target executable
add_executable(firmware.elf ${SOURCES})

# Math library linking
target_link_libraries(firmware.elf m)

# Output binary information
message(STATUS "Configured EdgeAI ECG Firmware for Cortex-M4 / Renode simulation.")
