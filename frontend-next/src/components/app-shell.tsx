"use client";

import { usePathname } from "next/navigation";
import { useState } from "react";
import { X } from "lucide-react";
import { Sidebar } from "./layout/sidebar";
import { Header } from "./layout/header";
import { Footer } from "./layout/footer";

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  // If we are on the chat page, we often want a specialized full-height layout.
  // We will keep the Sidebar, but not the Footer, so it feels like a real chat app.
  const isChat = pathname.startsWith("/chat");

  return (
    <div className="flex min-h-screen flex-col transparent">
      <div className="fixed top-4 bottom-4 left-4 z-30 hidden lg:block">
        <Sidebar />
      </div>
      
      {open ? (
        <div className="fixed inset-0 z-40 bg-slate-950/40 lg:hidden">
          <div className="h-full w-72 bg-white">
            <button
              aria-label="Close menu"
              className="absolute left-72 top-4 ml-3 rounded-md bg-white p-2 shadow"
              onClick={() => setOpen(false)}
            >
              <X className="h-5 w-5" />
            </button>
            <Sidebar setOpen={setOpen} />
          </div>
        </div>
      ) : null}

      <div className="flex min-h-screen flex-col lg:pl-[104px]">
        {!isChat && <Header setOpen={setOpen} />}
        
        {/* If it's chat, the chat page itself handles its own header/layout */}
        <main className={`flex-1 ${!isChat ? "mx-auto w-full max-w-7xl p-4 md:p-6" : ""}`}>
          {children}
        </main>

        {!isChat && <Footer />}
      </div>
    </div>
  );
}
