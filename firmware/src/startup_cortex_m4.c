/**
 * @file startup_cortex_m4.c
 * @brief Minimal bare-metal startup code and vector table for Cortex-M4 / Renode.
 */

#include <stdint.h>

extern uint32_t _estack;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sidata;
extern uint32_t _sbss;
extern uint32_t _ebss;

extern int main(void);
extern void uart_init(void);

void Reset_Handler(void);
void Default_Handler(void) {
    while (1);
}

/* Cortex-M4 Exception Handlers */
void NMI_Handler(void)        __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void)  __attribute__((weak, alias("Default_Handler")));
void MemManage_Handler(void)  __attribute__((weak, alias("Default_Handler")));
void BusFault_Handler(void)   __attribute__((weak, alias("Default_Handler")));
void UsageFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SVC_Handler(void)        __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void)     __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void)    __attribute__((weak, alias("Default_Handler")));

/* Vector Table placed at beginning of Flash (0x08000000) */
__attribute__((section(".isr_vector"), used))
const uint32_t* const g_pfnVectors[] = {
    (uint32_t*)&_estack,        /* Top of Stack (0x20040000 = 256 KB SRAM) */
    (uint32_t*)Reset_Handler,   /* Reset Handler */
    (uint32_t*)NMI_Handler,
    (uint32_t*)HardFault_Handler,
    (uint32_t*)MemManage_Handler,
    (uint32_t*)BusFault_Handler,
    (uint32_t*)UsageFault_Handler,
    0, 0, 0, 0,                 /* Reserved */
    (uint32_t*)SVC_Handler,
    0, 0,                       /* Reserved */
    (uint32_t*)PendSV_Handler,
    (uint32_t*)SysTick_Handler,
};

void Reset_Handler(void) {
    /* Copy data segment initializers from Flash to SRAM */
    uint32_t *pSrc = &_sidata;
    uint32_t *pDest = &_sdata;
    while (pDest < &_edata) {
        *pDest++ = *pSrc++;
    }

    /* Zero fill the bss segment in SRAM */
    uint32_t *pBss = &_sbss;
    while (pBss < &_ebss) {
        *pBss++ = 0;
    }

    /* Enable ARM Cortex-M4 FPU (Coprocessors CP10 and CP11 full access) */
    volatile uint32_t *cpacr = (volatile uint32_t *)0xE000ED88;
    *cpacr |= (0xFu << 20);
    __asm volatile("dsb\n\tisb");

    /* Initialize UART hardware before main */
    uart_init();

    /* Call main application */
    main();

    /* Hang if main returns */
    while (1);
}
