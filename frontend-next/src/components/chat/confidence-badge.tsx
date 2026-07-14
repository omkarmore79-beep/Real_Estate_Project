import { CheckCircle2, Sparkles, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export function ConfidenceBadge({ level }: { level: "high" | "medium" | "low" }) {
  const cfg = {
    high: { label: "High confidence", icon: CheckCircle2, cls: "bg-emerald-50 text-emerald-700 border-emerald-200" },
    medium: { label: "Medium confidence", icon: Sparkles, cls: "bg-amber-50 text-amber-700 border-amber-200" },
    low: { label: "Low confidence", icon: XCircle, cls: "bg-red-50 text-red-700 border-red-200" },
  };
  const { label, icon: Icon, cls } = cfg[level];
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium", cls)}>
      <Icon className="h-3 w-3" />
      {label}
    </span>
  );
}
