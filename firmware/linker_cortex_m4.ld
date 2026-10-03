/* Linker script for ARM Cortex-M4 Edge-AI ECG Node (256 KB SRAM, 1 MB Flash) */

ENTRY(Reset_Handler)

/* Memory layout matching Review-1 specifications & Renode cortex_m4_ecg.repl */
MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K /* 1.0 MB Flash */
  SRAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 256K  /* 256 KB SRAM */
}

_estack = ORIGIN(SRAM) + LENGTH(SRAM); /* End of SRAM (Top of stack) */

SECTIONS
{
  /* Vector table at start of Flash */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } >FLASH

  /* Program code and constants */
  .text :
  {
    . = ALIGN(4);
    *(.text)
    *(.text*)
    *(.glue_7)
    *(.glue_7t)
    *(.eh_frame)
    KEEP (*(.init))
    KEEP (*(.fini))
    . = ALIGN(4);
    _etext = .;
  } >FLASH

  /* Read-only data (includes INT8 quantized student model weights) */
  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } >FLASH

  /* Initialized data placed in SRAM, stored in Flash */
  _sidata = LOADADDR(.data);
  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data)
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } >SRAM AT> FLASH

  /* Uninitialized data in SRAM */
  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
    _end = .;
    __end__ = .;
    end = .;
  } >SRAM
}
