import Link from "next/link";
import { AlertCircle, Clock } from "lucide-react";
import { cn } from "@/lib/utils";

export type DocumentStatus = {
  status: string;
  progress: number;
  message: string;
  text_chunks_indexed?: number;
  images_indexed?: number;
  total_pages?: number;
};

export function ProcessingBanner({ status, documentId }: { status: DocumentStatus | null; documentId: string }) {
  if (!status || status.status === "ready") return null;
  const isFailed = status.status === "failed";

  return (
    <div className={cn(
      "mx-4 mt-3 rounded-lg border p-3",
      isFailed ? "border-red-200 bg-red-50" : "border-amber-200 bg-amber-50",
    )}>
      <div className="flex items-center gap-2">
        {isFailed
          ? <AlertCircle className="h-4 w-4 text-red-500 shrink-0" />
          : <Clock className="h-4 w-4 text-amber-600 shrink-0 animate-pulse" />
        }
        <p className="text-sm font-medium text-foreground">
          {isFailed
            ? "Document processing failed — try re-uploading."
            : `Indexing in progress (${status.progress}%) — ${status.message}`
          }
        </p>
        {!isFailed && (
          <Link
            href={`/documents/upload`}
            className="ml-auto shrink-0 text-xs text-amber-700 underline"
          >
            View status
          </Link>
        )}
      </div>
      {!isFailed && (
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/60">
          <div
            className="h-full rounded-full bg-amber-500 transition-all duration-500"
            style={{ width: `${status.progress}%` }}
          />
        </div>
      )}
    </div>
  );
}
