import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fourth Down Forecast | NFL Predictions",
  description:
    "Transparent, versioned NFL score predictions from a two-stage machine learning model.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
