/**
 * @file uart_stm32.h
 * @brief Minimal STM32 UART driver for Renode simulation console output.
 */

#ifndef UART_STM32_H_
#define UART_STM32_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* STM32 USART1 base address on Cortex-M4 APB2 bus */
#define USART1_BASE 0x40011000

/* Register offsets */
#define USART_SR_OFFSET  0x00
#define USART_DR_OFFSET  0x04
#define USART_CR1_OFFSET 0x0C

/* Function prototypes */
void uart_init(void);
void uart_putc(char c);
void uart_puts(const char* str);

#ifdef __cplusplus
}
#endif

#endif /* UART_STM32_H_ */
