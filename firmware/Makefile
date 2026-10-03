# Makefile for EdgeAI ECG Cortex-M4 Firmware targeting Renode

CC = arm-none-eabi-gcc
OBJCOPY = arm-none-eabi-objcopy
SIZE = arm-none-eabi-size

TARGET = build/firmware.elf
BIN = build/firmware.bin

CFLAGS = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 \
         -Wall -Wextra -ffunction-sections -fdata-sections \
         -Iinc

LDFLAGS = -Tlinker_cortex_m4.ld -Wl,--gc-sections --specs=nosys.specs -lm

SRCS = src/startup_cortex_m4.c \
       src/uart_stm32.c \
       src/main.c \
       src/ecg_dsp.c \
       src/state_scheduler.c \
       src/secure_telemetry.c

OBJS = $(SRCS:src/%.c=build/%.o)

.PHONY: all clean

all: $(TARGET) $(BIN)
	@$(SIZE) $(TARGET)

build/%.o: src/%.c | build
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(BIN): $(TARGET)
	$(OBJCOPY) -O binary $< $@

build:
	mkdir -p build

clean:
	rm -rf build
