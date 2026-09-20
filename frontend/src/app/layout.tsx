import type { Metadata } from "next";
import { Plus_Jakarta_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import SmoothScroll from "@/components/SmoothScroll";

const sansFont = Plus_Jakarta_Sans({
  variable: "--font-sans-main",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const monoFont = JetBrains_Mono({
  variable: "--font-mono-main",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "EdgeAI-ECG — Real-Time Privacy-Preserving Cardiac Edge Pipeline",
  description:
    "An ultra-low-power edge-AI monitoring pipeline that filters 99% of normal cardiac rhythms directly on-device, waking its radio transceiver only when critical arrhythmias occur.",
  keywords: ["Edge AI", "TinyML", "ECG Monitoring", "Privacy Preserving", "AES-GCM", "Healthcare IoT"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${sansFont.variable} ${monoFont.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-base text-ink">
        <SmoothScroll>{children}</SmoothScroll>
      </body>
    </html>
  );
}
