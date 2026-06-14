import type { Metadata } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import "./globals.css";

export const metadata: Metadata = {
  title: "ChinaRisk by AgentSafe — China Risk Intelligence",
  description:
    "China risk intelligence for institutional investors. Mandarin + English source analysis across prediction markets, news signals, and Chinese-language feeds.",
  keywords: [
    "China risk",
    "geopolitical risk",
    "US-China",
    "trade policy",
    "financial intelligence",
    "risk assessment",
    "PBOC",
    "Taiwan strait",
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
        <body className="antialiased" suppressHydrationWarning>{children}</body>
      </html>
    </ClerkProvider>
  );
}
