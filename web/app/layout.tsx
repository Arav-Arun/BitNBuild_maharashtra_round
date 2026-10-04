import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "../components/shell/AppShell";
import { ShapeGrid } from "../components/ui/ShapeGrid";

export const metadata: Metadata = { title: "Black Box · Agent debugger", description: "Failure localization and selective replay for AI agents" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><ShapeGrid speed={0}squareSize={44} borderColor="rgba(0,0,0,0.14)" hoverFillColor="#ebebeb" hoverTrailAmount={4} /><AppShell>{children}</AppShell></body></html>;
}
