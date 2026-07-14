import { BookOpen, FileText } from "lucide-react";
import type { Citation } from "@/lib/backend-data";

export function CitationsPanel({ citations }: { citations: Citation[] }) {
  if (!citations?.length) return null;
  return (
    <div className="mt-3 rounded-md border border-blue-100 bg-blue-50/60 p-3">
      <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-blue-700">
        <BookOpen className="h-3.5 w-3.5" /> Sources
      </p>
      <div className="space-y-1.5">
        {citations.map((c, i) => (
          <div key={c.citation_id ?? i} className="rounded bg-white/80 px-2.5 py-1.5 text-xs">
            <div className="flex items-center gap-2 font-medium text-foreground">
              <FileText className="h-3 w-3 text-blue-500 shrink-0" />
              <span className="truncate">{c.source_file || c.document_id || "Document"}</span>
              {c.ocr_used && (
                <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold text-amber-800 shrink-0">
                  OCR
                </span>
              )}
              {c.page_number != null && (
                <span className="ml-auto shrink-0 text-muted-foreground">p.{c.page_number}</span>
              )}
            </div>
            {c.snippet && (
              <p className="mt-1 line-clamp-2 text-muted-foreground">{c.snippet}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
