import type { Metadata } from "next";
import "./globals.css";
import Navbar from "./components/navbar";

export const metadata: Metadata = {
  title: "GraphShield",
  description: "Graph machine learning research platform for illicit transaction detection",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <div className="gs-app-shell">
          <Navbar />

          <main className="gs-main-content">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}