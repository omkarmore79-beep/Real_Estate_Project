"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Building2,
  BotMessageSquare,
  FileSearch,
  Files,
  Home,
  Menu,
  Plus,
  Search,
  Settings,
  Upload,
  X,
} from "lucide-react";
import { useState } from "react";
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

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  if (pathname.startsWith("/chat")) {
    return <div className="min-h-screen bg-background">{children}</div>;
  }

  const sidebar = (
    <aside className="flex h-full w-72 flex-col border-r bg-white">
      <div className="flex h-16 items-center gap-3 border-b px-5">
        <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-primary-foreground">
          <Building2 className="h-5 w-5" />
        </div>
        <div>
          <p className="text-sm font-semibold">EstateDocs</p>
          <p className="text-xs text-muted-foreground">Document intelligence</p>
        </div>
      </div>
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navigation.map((item) => {
          const Icon = item.icon;
          const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setOpen(false)}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition hover:bg-secondary hover:text-foreground",
                active && "bg-primary/10 text-primary",
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="border-t p-4">
        <Link
          href="/projects/new"
          className="flex h-10 items-center justify-center gap-2 rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground transition hover:bg-primary/90"
        >
          <Plus className="h-4 w-4" />
          New project
        </Link>
      </div>
    </aside>
  );

  return (
    <div className="min-h-screen bg-background">
      <div className="fixed inset-y-0 left-0 z-30 hidden lg:block">{sidebar}</div>
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
            {sidebar}
          </div>
        </div>
      ) : null}
      <div className="lg:pl-72">
        <header className="sticky top-0 z-20 flex h-16 items-center gap-4 border-b bg-white/90 px-4 backdrop-blur md:px-6">
          <button
            aria-label="Open menu"
            className="rounded-md border p-2 lg:hidden"
            onClick={() => setOpen(true)}
          >
            <Menu className="h-5 w-5" />
          </button>
          <div className="relative max-w-xl flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              className="field pl-9"
              placeholder="Search builder, project, RERA, document type..."
            />
          </div>
          <Link
            href="/documents/upload"
            className="hidden h-10 items-center gap-2 rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground md:flex"
          >
            <Upload className="h-4 w-4" />
            Upload document
          </Link>
        </header>
        <main className="mx-auto max-w-7xl px-4 py-6 md:px-6">{children}</main>
      </div>
    </div>
  );
}
