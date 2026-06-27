"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { Eye, MessageSquareText, Trash2 } from "lucide-react";

export function DocumentActions({ documentId }: { documentId?: string }) {
  const router = useRouter();

  if (!documentId) return null;

  async function deleteDocument() {
    if (!documentId) return;

    const confirmed = window.confirm("Delete this PDF and its extracted chatbot data?");
    if (!confirmed) return;

    const response = await fetch(`/api/documents/${documentId}`, {
      method: "DELETE",
    });

    if (!response.ok) {
      const data = await response.json().catch(() => null);
      window.alert(data?.error ?? "Could not delete this document.");
      return;
    }

    if (localStorage.getItem("activeDocumentId") === documentId) {
      localStorage.removeItem("activeDocumentId");
    }

    router.refresh();
  }

  return (
    <div className="flex justify-end gap-2">
      <Link
        href={`/chat?documentId=${documentId}`}
        className="rounded-md border p-2 text-muted-foreground transition hover:bg-secondary hover:text-foreground"
        aria-label="Chat with this PDF"
        onClick={() => localStorage.setItem("activeDocumentId", documentId)}
      >
        <MessageSquareText className="h-4 w-4" />
      </Link>
      <Link
        href={`/api/documents/${documentId}`}
        target="_blank"
        className="rounded-md border p-2 text-muted-foreground transition hover:bg-secondary hover:text-foreground"
        aria-label="View PDF"
      >
        <Eye className="h-4 w-4" />
      </Link>
      <button
        type="button"
        onClick={deleteDocument}
        className="rounded-md border p-2 text-muted-foreground transition hover:border-red-200 hover:bg-red-50 hover:text-red-700"
        aria-label="Delete PDF"
      >
        <Trash2 className="h-4 w-4" />
      </button>
    </div>
  );
}
