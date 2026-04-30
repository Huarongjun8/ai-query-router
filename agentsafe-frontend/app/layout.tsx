import type { Metadata } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import "./globals.css";

export const metadata: Metadata = {
  title: "AgentSafe — Geopolitical Risk Intelligence",
  description:
    "Multi-agent geopolitical risk intelligence platform for financial services. Real-time US-China trade and technology policy signals.",
  keywords: [
    "geopolitical risk",
    "US-China",
    "trade policy",
    "financial intelligence",
    "risk assessment",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ClerkProvider>
      <html lang="en" className="dark">
        <body className="antialiased">{children}</body>
      </html>
    </ClerkProvider>
  );
}
