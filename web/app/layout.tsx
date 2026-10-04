import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "../components/shell/AppShell";

export const metadata: Metadata = { title: "Black Box · Agent debugger", description: "Failure localization and selective replay for AI agents" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><AppShell>{children}</AppShell></body></html>;
}
