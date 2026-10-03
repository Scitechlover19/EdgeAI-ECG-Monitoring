/**
 * @file uart_stm32.c
 * @brief STM32 UART implementation and newlib _write retargeting for Renode.
 */

#include "uart_stm32.h"

#define REG32(addr) (*(volatile uint32_t*)(addr))
typedef uint32_t volatile_reg32;

void uart_init(void) {
    /* Enable USART1 clock in RCC APB2ENR (0x40023844 bit 4) */
    volatile uint32_t* apb2enr = (volatile uint32_t*)0x40023844;
    *apb2enr |= (1u << 4);

    /* Enable UART and Transmitter in CR1 */
    volatile uint32_t* cr1 = (volatile uint32_t*)(USART1_BASE + USART_CR1_OFFSET);
    *cr1 |= 0x2008; /* UE (bit 13) | TE (bit 3) */
}

void uart_putc(char c) {
    volatile uint32_t* dr = (volatile uint32_t*)(USART1_BASE + USART_DR_OFFSET);
    *dr = (uint32_t)c;
}

void uart_puts(const char* str) {
    if (!str) return;
    while (*str) {
        if (*str == '\n') {
            uart_putc('\r');
        }
        uart_putc(*str++);
    }
}

/* Retarget standard C library printf / _write to UART */
int _write(int file, char *ptr, int len) {
    (void)file;
    for (int i = 0; i < len; i++) {
        if (ptr[i] == '\n') {
            uart_putc('\r');
        }
        uart_putc(ptr[i]);
    }
    return len;
}
