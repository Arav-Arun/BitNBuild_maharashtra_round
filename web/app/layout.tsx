import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Black Box",
  description: "Failure localization and selective replay for AI agents",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
