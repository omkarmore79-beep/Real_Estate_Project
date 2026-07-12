"use client";

import Link from "next/link";
import {
  AlertCircle,
  ArrowLeft,
  BookOpen,
  BotMessageSquare,
  CheckCircle2,
  Clock,
  FileText,
  Home,
  ImageIcon,
  Loader2,
  Send,
  Sparkles,
  XCircle,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui";
import {
  getBuilderName,
  getProjectTitle,
  resolveImageUrl,
  type Citation,
  type ImageResult,
  type UploadedProject,
} from "@/lib/backend-data";
import { cn } from "@/lib/utils";

// ── Types ──────────────────────────────────────────────────────────────────────
type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
  images?: string[];
  imageResults?: ImageResult[];
  citations?: Citation[];
  confidence?: "high" | "medium" | "low";
  isProcessingWarning?: boolean;
};

type DocumentStatus = {
  status: string;
  progress: number;
  message: string;
  text_chunks_indexed?: number;
  images_indexed?: number;
  total_pages?: number;
};

const QUICK_QUESTIONS = [
  "What is the project name?",
  "Where is the project located?",
  "What amenities are available?",
  "Show me the floor plan",
  "What is the RERA number?",
  "Show me the location plan",
  "What is the possession date?",
  "What are the contact details?",
];

const BACKEND_URL = "http://127.0.0.1:8000";

// ── Confidence badge ───────────────────────────────────────────────────────────
function ConfidenceBadge({ level }: { level: "high" | "medium" | "low" }) {
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

// ── Citations panel ────────────────────────────────────────────────────────────
function CitationsPanel({ citations }: { citations: Citation[] }) {
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

// ── Image gallery ──────────────────────────────────────────────────────────────
function ImageGallery({ images, imageResults }: { images?: string[]; imageResults?: ImageResult[] }) {
  type DisplayImage = { url: string; caption?: string; page?: number | null; type?: string };
  const items: DisplayImage[] = [];

  if (imageResults?.length) {
    for (const img of imageResults) {
      const url = resolveImageUrl(img, BACKEND_URL);
      if (url) items.push({ url, caption: img.caption, page: img.page_number, type: img.image_type });
    }
  }
  if (images?.length && items.length === 0) {
    for (const img of images) {
      const url = resolveImageUrl(img, BACKEND_URL);
      if (url) items.push({ url });
    }
  }
  if (!items.length) return null;

  return (
    <div className="mt-3 grid gap-3 sm:grid-cols-2">
      {items.map((item, i) => (
        <div key={i} className="overflow-hidden rounded-md border bg-white shadow-sm">
          <img
            src={item.url}
            alt={item.caption || `Page image ${i + 1}`}
            className="max-h-72 w-full object-contain"
            loading="lazy"
          />
          {(item.caption || item.type) && (
            <div className="border-t bg-secondary/50 px-2.5 py-1.5">
              {item.type && (
                <p className="text-xs font-medium capitalize text-primary">
                  {item.type.replace(/_/g, " ")}{item.page ? ` — p.${item.page}` : ""}
                </p>
              )}
              {item.caption && (
                <p className="line-clamp-2 text-xs text-muted-foreground">{item.caption}</p>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Processing banner ──────────────────────────────────────────────────────────
function ProcessingBanner({ status, documentId }: { status: DocumentStatus | null; documentId: string }) {
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

// ── Main component ─────────────────────────────────────────────────────────────
export default function UserChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 1,
      role: "assistant",
      content:
        "Hi, I can answer questions from uploaded real estate brochures using Hybrid Multimodal RAG. Select a document and ask about amenities, floor plans, RERA, pricing, possession dates, or location — I'll cite my sources.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [activeDocumentId, setActiveDocumentId] = useState("");
  const [documents, setDocuments] = useState<UploadedProject[]>([]);
  const [includeImages, setIncludeImages] = useState(true);
  const [docStatus, setDocStatus] = useState<DocumentStatus | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  // ── Load documents on mount ──────────────────────────────────────────────────
  useEffect(() => {
    const documentIdFromUrl = new URLSearchParams(window.location.search).get("documentId");
    const storedId = localStorage.getItem("activeDocumentId") || "";
    const nextId = documentIdFromUrl || storedId;
    setActiveDocumentId(nextId);
    if (nextId) localStorage.setItem("activeDocumentId", nextId);

    fetch("/api/projects")
      .then((r) => r.json())
      .then((data) => setDocuments(Array.isArray(data.projects) ? data.projects : []))
      .catch(() => setDocuments([]));
  }, []);

  // ── Load / Save chat history from localStorage ────────────────────────────────
  useEffect(() => {
    if (!activeDocumentId) {
      setMessages([
        {
          id: 1,
          role: "assistant",
          content:
            "Hi, I can answer questions from uploaded real estate brochures using Hybrid Multimodal RAG. Select a document and ask about amenities, floor plans, RERA, pricing, possession dates, or location — I'll cite my sources.",
        },
      ]);
      return;
    }

    const stored = localStorage.getItem(`chat_history_${activeDocumentId}`);
    if (stored) {
      try {
        setMessages(JSON.parse(stored));
      } catch (err) {
        console.error("Failed to parse stored chat history:", err);
      }
    } else {
      setMessages([
        {
          id: 1,
          role: "assistant",
          content:
            "Hi, I can answer questions from uploaded real estate brochures using Hybrid Multimodal RAG. Select a document and ask about amenities, floor plans, RERA, pricing, possession dates, or location — I'll cite my sources.",
        },
      ]);
    }
  }, [activeDocumentId]);

  useEffect(() => {
    if (activeDocumentId && messages.length > 0) {
      localStorage.setItem(`chat_history_${activeDocumentId}`, JSON.stringify(messages));
    }
  }, [messages, activeDocumentId]);

  // ── Poll document status ─────────────────────────────────────────────────────
  const pollStatus = useCallback(async (docId: string) => {
    if (!docId) return;
    try {
      const res = await fetch(`/api/documents/${docId}/status`, { cache: "no-store" });
      if (!res.ok) return;
      const data: DocumentStatus = await res.json();
      setDocStatus(data);
      if (data.status !== "ready" && data.status !== "failed") {
        pollRef.current = setTimeout(() => pollStatus(docId), 3000);
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    if (!activeDocumentId) {
      setDocStatus(null);
      return;
    }
    if (pollRef.current) clearTimeout(pollRef.current);
    pollStatus(activeDocumentId);
    return () => { if (pollRef.current) clearTimeout(pollRef.current); };
  }, [activeDocumentId, pollStatus]);

  // ── Auto-scroll ──────────────────────────────────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const activeDocument = documents.find((d) => d.document_id === activeDocumentId);
  const isDocReady = activeDocumentId !== "" && docStatus?.status === "ready";


  function handleDocumentChange(documentId: string) {
    setActiveDocumentId(documentId);
    if (documentId) {
      localStorage.setItem("activeDocumentId", documentId);
      window.history.replaceState(null, "", `/chat?documentId=${documentId}`);
    } else {
      localStorage.removeItem("activeDocumentId");
      window.history.replaceState(null, "", "/chat");
      setDocStatus(null);
    }
  }

  async function sendMessage(messageText: string) {
    const clean = messageText.trim();
    if (!clean || isLoading) return;

    setMessages((c) => [...c, { id: Date.now(), role: "user", content: clean }]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: clean,
          documentId: activeDocumentId,
          document_id: activeDocumentId,
          include_images: includeImages,
          top_k: 8,
        }),
      });
      const data = await res.json();

      if (res.ok) {
        const rawImages: unknown[] = Array.isArray(data.images) ? data.images : [];
        const imageResults = rawImages.filter(
          (img) => typeof img === "object" && img !== null && "image_url" in (img as object),
        ) as ImageResult[];
        const legacyImages = rawImages.filter((img) => typeof img === "string") as string[];

        setMessages((c) => [
          ...c,
          {
            id: Date.now() + 1,
            role: "assistant",
            content: data.answer || "No answer returned.",
            images: legacyImages,
            imageResults: imageResults.length > 0 ? imageResults : undefined,
            citations: Array.isArray(data.citations) ? data.citations : undefined,
            confidence: data.confidence,
            isProcessingWarning: data.status === "processing",
          },
        ]);
      } else {
        setMessages((c) => [
          ...c,
          {
            id: Date.now() + 1,
            role: "assistant",
            content: data.error || data.answer || "An error occurred.",
          },
        ]);
      }
    } catch {
      setMessages((c) => [
        ...c,
        { id: Date.now() + 1, role: "assistant", content: "Could not reach the backend. Is FastAPI running?" },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    sendMessage(input);
  }

  return (
    <main className="min-h-screen bg-[linear-gradient(180deg,#f8fafc_0%,#eef7f5_48%,#f8fafc_100%)]">
      <header className="border-b bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-4 md:px-6">
          <Link href="/" className="inline-flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-primary">
            <ArrowLeft className="h-4 w-4" />
            Admin panel
          </Link>
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <BotMessageSquare className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold">Estate Assistant</p>
              <p className="text-xs text-muted-foreground">Hybrid Multimodal RAG · Grounded answers</p>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-6 px-4 py-6 md:px-6 lg:grid-cols-[360px_1fr]">
        {/* ── Sidebar ─────────────────────────────────────────────────────── */}
        <aside className="space-y-4">
          <section className="panel overflow-hidden">
            <div className="bg-primary p-5 text-primary-foreground">
              <div className="flex items-center gap-3">
                <FileText className="h-6 w-6" />
                <div>
                  <h1 className="text-xl font-semibold">Find your next home</h1>
                  <p className="mt-1 text-sm text-primary-foreground/80">Ask the brochure in plain language.</p>
                </div>
              </div>
            </div>
            <div className="space-y-4 p-5">
              <div>
                <p className="text-sm text-muted-foreground">Active document</p>
                <h2 className="mt-1 font-semibold">
                  {activeDocument ? getProjectTitle(activeDocument) : "No document selected"}
                </h2>
                {activeDocument && (
                  <p className="mt-1 text-xs text-muted-foreground">
                    {getBuilderName(activeDocument)} · {activeDocument.source_file}
                  </p>
                )}
                {/* Document readiness indicator */}
                {activeDocumentId && docStatus && (
                  <div className={cn(
                    "mt-2 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium",
                    docStatus.status === "ready"
                      ? "bg-emerald-100 text-emerald-700"
                      : docStatus.status === "failed"
                      ? "bg-red-100 text-red-700"
                      : "bg-amber-100 text-amber-700",
                  )}>
                    {docStatus.status === "ready"
                      ? <CheckCircle2 className="h-3 w-3" />
                      : docStatus.status === "failed"
                      ? <XCircle className="h-3 w-3" />
                      : <Loader2 className="h-3 w-3 animate-spin" />
                    }
                    {docStatus.status === "ready"
                      ? "Ready for chat"
                      : docStatus.status === "failed"
                      ? "Processing failed"
                      : `Indexing… ${docStatus.progress}%`
                    }
                  </div>
                )}
              </div>

              <label className="space-y-2">
                <span className="label">Ask from PDF</span>
                <select
                  className="field"
                  value={activeDocumentId}
                  onChange={(e) => handleDocumentChange(e.target.value)}
                >
                  <option value="">Select a PDF / project</option>
                  {documents.map((doc, i) => (
                    <option
                      key={doc.document_id ?? `${doc.source_file}-${i}`}
                      value={doc.document_id ?? ""}
                    >
                      {getBuilderName(doc)} / {getProjectTitle(doc)}
                    </option>
                  ))}
                </select>
              </label>

              <label className="flex cursor-pointer items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="rounded"
                  checked={includeImages}
                  onChange={(e) => setIncludeImages(e.target.checked)}
                />
                Include images in answers
              </label>

              <div className="flex flex-wrap gap-2">
                <Badge>{activeDocumentId ? "Real upload" : "Awaiting upload"}</Badge>
                <Badge>Grounded answers</Badge>
                <Badge>Hybrid RAG</Badge>
              </div>

              <div className="grid gap-2 text-sm">
                <div className="flex items-center gap-2 rounded-md bg-secondary px-3 py-2">
                  <Home className="h-4 w-4 text-primary" />
                  Project details, towers, and location
                </div>
                <div className="flex items-center gap-2 rounded-md bg-secondary px-3 py-2">
                  <ImageIcon className="h-4 w-4 text-primary" />
                  Floor plans and brochure images
                </div>
                <div className="flex items-center gap-2 rounded-md bg-secondary px-3 py-2">
                  <Sparkles className="h-4 w-4 text-primary" />
                  AI answers with source citations
                </div>
              </div>
            </div>
          </section>

          <section className="panel p-5">
            <h2 className="font-semibold">Quick questions</h2>
            <div className="mt-4 grid gap-2">
              {QUICK_QUESTIONS.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => sendMessage(q)}
                  className="rounded-md border bg-white px-3 py-2 text-left text-sm transition hover:border-primary/40 hover:bg-primary/5 disabled:opacity-50"
                  disabled={isLoading || !isDocReady}
                >
                  {q}
                </button>
              ))}
            </div>
          </section>
        </aside>

        {/* ── Chat area ─────────────────────────────────────────────────────── */}
        <section className="panel flex min-h-[calc(100vh-120px)] flex-col overflow-hidden">
          <div className="border-b bg-white px-5 py-4">
            <h2 className="font-semibold">Chat with AI</h2>
            <p className="text-sm text-muted-foreground">
              Answers come from retrieved document chunks. Sources are cited below each response.
            </p>
          </div>

          {/* Processing warning banner */}
          {activeDocumentId && docStatus && docStatus.status !== "ready" && (
            <ProcessingBanner status={docStatus} documentId={activeDocumentId} />
          )}

          <div className="flex-1 space-y-4 overflow-y-auto bg-secondary/40 p-4 md:p-5">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={cn("flex", msg.role === "user" ? "justify-end" : "justify-start")}
              >
                <div
                  className={cn(
                    "max-w-[820px] rounded-lg px-4 py-3 text-sm shadow-sm",
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : msg.isProcessingWarning
                      ? "border border-amber-200 bg-amber-50"
                      : "border bg-white text-foreground",
                  )}
                >
                  {msg.role === "assistant" && msg.confidence && (
                    <div className="mb-2">
                      <ConfidenceBadge level={msg.confidence} />
                    </div>
                  )}
                  <p className="whitespace-pre-wrap leading-6">{msg.content}</p>
                  <ImageGallery images={msg.images} imageResults={msg.imageResults} />
                  {msg.role === "assistant" && <CitationsPanel citations={msg.citations ?? []} />}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="inline-flex items-center gap-2 rounded-lg border bg-white px-4 py-3 text-sm text-muted-foreground shadow-sm">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Searching documents and generating grounded answer…
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="border-t bg-white p-4">
            {!isDocReady && activeDocumentId && (
              <p className="mb-2 flex items-center gap-1.5 text-xs text-amber-700">
                <Clock className="h-3.5 w-3.5" />
                Document is still indexing. Chat will be enabled once indexing completes.
              </p>
            )}
            <div className="flex gap-3">
              <input
                className="field h-11"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={!isDocReady}
                placeholder={
                  !activeDocumentId
                    ? "Select a document from the sidebar to start chatting..."
                    : !isDocReady
                    ? `Indexing document (${docStatus?.progress ?? 0}%) — please wait…`
                    : "Ask about price, floor plan, amenities, RERA, possession…"
                }
              />
              <button
                type="submit"
                className="inline-flex h-11 items-center gap-2 rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-60"
                disabled={isLoading || !isDocReady || input.trim().length === 0}
              >
                <Send className="h-4 w-4" />
                Send
              </button>
            </div>
          </form>

        </section>
      </div>
    </main>
  );
}
