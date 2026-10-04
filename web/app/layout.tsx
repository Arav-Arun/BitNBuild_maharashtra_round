import type { Metadata } from "next";
import { Bricolage_Grotesque, Instrument_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { AppShell } from "../components/shell/AppShell";
import { ShapeGrid } from "../components/ui/ShapeGrid";

const display = Bricolage_Grotesque({ subsets: ["latin"], variable: "--font-display", axes: ["opsz", "wdth"] });
const sans = Instrument_Sans({ subsets: ["latin"], variable: "--font-sans", axes: ["wdth"] });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Black Box · Agent debugger",
  description: "Find the step that broke an AI agent's run, and prove the fix with paired replays.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${sans.variable} ${mono.variable}`}>
      <body>
        <ShapeGrid speed={0} squareSize={44} borderColor="rgba(0,0,0,0.14)" hoverFillColor="#ebebeb" hoverTrailAmount={4} />
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
