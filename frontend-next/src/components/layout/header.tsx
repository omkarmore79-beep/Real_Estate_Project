"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import Cookies from "js-cookie";
import { Bell, ChevronDown, Menu, Search, Upload, UserCircle, Building2, Tractor, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

export function Header({ setOpen }: { setOpen: (open: boolean) => void }) {
  const pathname = usePathname();
  const router = useRouter();
  const [domain, setDomain] = useState("real-estate");
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  useEffect(() => {
    const savedDomain = Cookies.get("domain");
    if (savedDomain) setDomain(savedDomain);
  }, []);

  const handleDomainChange = (newDomain: string) => {
    setDomain(newDomain);
    Cookies.set("domain", newDomain, { expires: 365, path: "/" });
    setIsDropdownOpen(false);
    router.refresh();
  };

  // Simple breadcrumb logic based on pathname
  const pathSegments = pathname.split("/").filter(Boolean);
  const currentPage = pathSegments.length > 0 
    ? pathSegments[pathSegments.length - 1].charAt(0).toUpperCase() + pathSegments[pathSegments.length - 1].slice(1)
    : "Dashboard";

  return (
    <header className="sticky top-4 z-20 mx-4 flex h-16 items-center gap-4 rounded-2xl border border-white/50 bg-white/40 px-4 backdrop-blur-md shadow-[0_8px_32px_rgba(0,0,0,0.05)] md:px-6">
      <button
        aria-label="Open menu"
        className="rounded-md border p-2 lg:hidden"
        onClick={() => setOpen(true)}
      >
        <Menu className="h-5 w-5" />
      </button>

      {/* Workspace Domain Toggle */}
      <div className="relative hidden md:block">
        <button 
          onClick={() => setIsDropdownOpen(!isDropdownOpen)}
          className="flex h-9 items-center gap-2 rounded-lg bg-white/50 px-3 text-sm font-medium text-slate-700 shadow-sm ring-1 ring-black/5 transition hover:bg-white/80"
        >
          {domain === "real-estate" ? <Building2 className="h-4 w-4 text-blue-500" /> : <Tractor className="h-4 w-4 text-orange-500" />}
          <span>{domain === "real-estate" ? "Real Estate" : "Heavy Machinery"}</span>
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        </button>

        {isDropdownOpen && (
          <div className="absolute left-0 top-11 z-50 w-48 rounded-xl border border-white/60 bg-white/90 p-1 backdrop-blur-xl shadow-lg ring-1 ring-black/5">
            <button 
              onClick={() => handleDomainChange("real-estate")}
              className={cn("flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm transition hover:bg-slate-100", domain === "real-estate" && "bg-slate-50 font-semibold text-blue-600")}
            >
              <Building2 className="h-4 w-4" /> Real Estate
            </button>
            <button 
              onClick={() => handleDomainChange("machinery")}
              className={cn("flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm transition hover:bg-slate-100", domain === "machinery" && "bg-slate-50 font-semibold text-orange-600")}
            >
              <Tractor className="h-4 w-4" /> Heavy Machinery
            </button>
          </div>
        )}
      </div>

      {/* Breadcrumbs */}
      <div className="hidden flex-1 md:flex items-center gap-2 text-sm text-muted-foreground ml-2">
        <Link href="/" className="hover:text-foreground">Home</Link>
        <span>/</span>
        <span className="font-medium text-foreground">{currentPage.replace("-", " ")}</span>
      </div>

      <div className="flex flex-1 md:flex-none items-center justify-end gap-3">
        {/* Command Palette Search Button */}
        <button className="group hidden h-9 w-64 items-center justify-between rounded-lg border border-slate-200 bg-white/50 px-3 text-sm text-muted-foreground shadow-sm transition hover:bg-white hover:border-slate-300 md:flex">
          <div className="flex items-center gap-2">
            <Search className="h-4 w-4 shrink-0 text-slate-400 group-hover:text-slate-500" />
            <span className="truncate">Search manuals...</span>
          </div>
          <kbd className="hidden rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-500 ring-1 ring-slate-200 sm:block">
            ⌘K
          </kbd>
        </button>

        <Link
          href="/documents/upload"
          className="hidden h-9 items-center gap-2 rounded-lg bg-primary px-4 text-sm font-semibold whitespace-nowrap text-primary-foreground shadow-sm transition hover:bg-primary/90 md:flex"
        >
          <Upload className="h-4 w-4 shrink-0" />
          Upload Document
        </Link>

        {/* System Status / Notifications */}
        <div className="flex items-center gap-1 border-l border-slate-200 pl-3 ml-1">
          <button className="relative flex h-9 w-9 items-center justify-center rounded-full text-slate-500 transition hover:bg-white/60 hover:text-slate-700" title="RAG System Status: Online">
            <Activity className="h-5 w-5 text-emerald-500" />
            <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-emerald-500 ring-2 ring-white/50 animate-pulse"></span>
          </button>

          <button className="relative flex h-9 w-9 items-center justify-center rounded-full text-slate-500 transition hover:bg-white/60 hover:text-slate-700">
            <Bell className="h-5 w-5" />
          </button>

          {/* Profile */}
          <Link href="/profile" className="flex h-9 w-9 items-center justify-center rounded-full transition hover:bg-white/60">
            <UserCircle className="h-7 w-7 text-slate-600" />
          </Link>
        </div>
      </div>
    </header>
  );
}
