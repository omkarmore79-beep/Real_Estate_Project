"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import Cookies from "js-cookie";
import {
  Building2,
  BotMessageSquare,
  FileSearch,
  Files,
  Home,
  Plus,
  Search,
  Settings,
  Upload,
  Database,
  User,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
  { href: "/", label: "Dashboard", icon: Home },
  { href: "/chat", label: "User chat", icon: BotMessageSquare },
  { href: "/builders", label: "Builders", icon: Building2 },
  { href: "/projects", label: "Projects", icon: Files },
  { href: "/documents", label: "Documents", icon: FileSearch },
  { href: "/documents/upload", label: "Upload", icon: Upload },
  { href: "/search", label: "Search", icon: Search },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar({ setOpen }: { setOpen?: (open: boolean) => void }) {
  const pathname = usePathname();
  const [domain, setDomain] = useState("real-estate");

  useEffect(() => {
    const savedDomain = Cookies.get("domain");
    if (savedDomain) setDomain(savedDomain);
  }, [pathname]); // Refresh when navigation happens

  return (
    <aside className="group flex h-full w-[72px] hover:w-72 flex-col overflow-hidden whitespace-nowrap rounded-2xl border border-white/50 bg-white/20 backdrop-blur-md shadow-[8px_8px_32px_rgba(0,0,0,0.05)] transition-all duration-300 ease-in-out">
      <div className="flex h-16 shrink-0 items-center gap-4 border-b border-white/30 px-[15px]">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground shadow-sm transition-transform group-hover:scale-105">
          <Database className="h-5 w-5" />
        </div>
        <div className="flex flex-col opacity-0 transition-opacity duration-300 group-hover:opacity-100">
          <p className="text-sm font-semibold text-slate-900">RAG Knowledge</p>
          <p className="text-xs text-muted-foreground">Document Intelligence</p>
        </div>
      </div>
      
      <nav className="flex-1 space-y-2 overflow-y-auto overflow-x-hidden px-3 py-4 scrollbar-hide">
        {navigation.map((item) => {
          const Icon = item.icon;
          const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
          
          let displayLabel = item.label;
          if (item.label === "Builders" && domain === "machinery") {
            displayLabel = "Manufacturers";
          }
          if (item.label === "Projects" && domain === "machinery") {
            displayLabel = "Manuals";
          }
          
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setOpen?.(false)}
              className={cn(
                "flex items-center gap-4 rounded-md px-3 py-2 text-sm font-medium text-slate-600 transition-all duration-300 hover:bg-white/80 hover:text-slate-900 hover:shadow-sm",
                active && "bg-white/90 text-primary shadow-sm font-semibold ring-1 ring-white/50",
              )}
            >
              <Icon className="h-5 w-5 shrink-0" />
              <span className="opacity-0 transition-opacity duration-300 group-hover:opacity-100">
                {displayLabel}
              </span>
            </Link>
          );
        })}
      </nav>

      <div className="shrink-0 border-t border-white/40 p-3">
        <Link
          href="/projects/new"
          className="mb-3 flex h-10 items-center gap-4 rounded-md bg-primary px-3 text-sm font-semibold text-primary-foreground shadow-sm transition-all hover:bg-primary/90"
        >
          <Plus className="h-5 w-5 shrink-0" />
          <span className="opacity-0 transition-opacity duration-300 group-hover:opacity-100">
            New project
          </span>
        </Link>
        
        {/* User Profile Badge */}
        <Link
          href="/profile"
          className="flex items-center gap-4 rounded-md px-2 py-2 transition-all duration-300 hover:bg-white/80 hover:shadow-sm"
        >
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary transition-transform duration-300 hover:scale-110">
            <User className="h-5 w-5" />
          </div>
          <div className="flex flex-1 flex-col overflow-hidden opacity-0 transition-opacity duration-300 group-hover:opacity-100">
            <p className="truncate text-sm font-medium text-slate-900">Admin User</p>
            <p className="truncate text-xs text-muted-foreground">admin@hyundai.com</p>
          </div>
        </Link>
      </div>
    </aside>
  );
}
