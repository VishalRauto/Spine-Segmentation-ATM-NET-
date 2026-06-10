import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Providers from "./providers";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ATM-Net++ | Lumbar Spine AI Diagnostics",
  description: "Anatomy-Aware Multimodal Lumbar Spine MRI Diagnostic and Segmentation System",
  keywords: ["spine MRI", "AI diagnostics", "lumbar spine", "segmentation", "radiology"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
