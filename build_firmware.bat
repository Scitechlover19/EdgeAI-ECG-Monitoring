@echo off
setlocal

:: Search for arm-none-eabi-gcc in common installation directories
where arm-none-eabi-gcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    if exist "C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi\14.2 rel1\bin" (
        set "PATH=%PATH%;C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi\14.2 rel1\bin"
    ) else if exist "C:\Program Files\Arm GNU Toolchain arm-none-eabi\14.2 rel1\bin" (
        set "PATH=%PATH%;C:\Program Files\Arm GNU Toolchain arm-none-eabi\14.2 rel1\bin"
    ) else if exist "C:\Program Files (x86)\GNU Arm Embedded Toolchain" (
        for /d %%D in ("C:\Program Files (x86)\GNU Arm Embedded Toolchain\*") do (
            if exist "%%D\bin" set "PATH=%PATH%;%%D\bin"
        )
    )
)

echo ========================================================
echo  Building EdgeAI ECG Cortex-M4 Firmware for Renode
echo ========================================================

if not exist firmware\build mkdir firmware\build

echo Compiling startup code...
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -Wall -Wextra -ffunction-sections -fdata-sections -Ifirmware\inc -c firmware\src\startup_cortex_m4.c -o firmware\build\startup_cortex_m4.o
if %ERRORLEVEL% neq 0 goto error

echo Compiling STM32 UART driver...
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -Wall -Wextra -ffunction-sections -fdata-sections -Ifirmware\inc -c firmware\src\uart_stm32.c -o firmware\build\uart_stm32.o
if %ERRORLEVEL% neq 0 goto error

echo Compiling ECG DSP filter...
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -Wall -Wextra -ffunction-sections -fdata-sections -Ifirmware\inc -c firmware\src\ecg_dsp.c -o firmware\build\ecg_dsp.o
if %ERRORLEVEL% neq 0 goto error

echo Compiling state scheduler...
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -Wall -Wextra -ffunction-sections -fdata-sections -Ifirmware\inc -c firmware\src\state_scheduler.c -o firmware\build\state_scheduler.o
if %ERRORLEVEL% neq 0 goto error

echo Compiling secure telemetry...
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -Wall -Wextra -ffunction-sections -fdata-sections -Ifirmware\inc -c firmware\src\secure_telemetry.c -o firmware\build\secure_telemetry.o
if %ERRORLEVEL% neq 0 goto error

echo Compiling main application with INT8 student model...
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -Wall -Wextra -ffunction-sections -fdata-sections -Ifirmware\inc -c firmware\src\main.c -o firmware\build\main.o
if %ERRORLEVEL% neq 0 goto error

echo Linking Cortex-M4 ELF binary...
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard firmware\build\startup_cortex_m4.o firmware\build\uart_stm32.o firmware\build\main.o firmware\build\ecg_dsp.o firmware\build\state_scheduler.o firmware\build\secure_telemetry.o -Tfirmware\linker_cortex_m4.ld -Wl,--gc-sections --specs=nosys.specs -lm -o firmware\build\firmware.elf
if %ERRORLEVEL% neq 0 goto error

echo.
echo ========================================================
echo  BUILD SUCCESSFUL: firmware\build\firmware.elf generated!
echo ========================================================
echo Now run this command in your Renode monitor window:
echo   s @d:\Project\EdgeAI-ECG-Monitoring\firmware\renode\simulate.resc
echo ========================================================
goto end

:error
echo.
echo [ERROR] Build failed! Please ensure arm-none-eabi-gcc is installed.

:end
