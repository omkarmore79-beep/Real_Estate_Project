"use client";

import Link from "next/link";
import { ArrowLeft, BotMessageSquare, CheckCircle2, Clock, FileText, Home, ImageIcon, Loader2, Send, Sparkles, XCircle } from "lucide-react";
import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui";
import { getBuilderName, getProjectTitle, type Citation, type ImageResult, type UploadedProject } from "@/lib/backend-data";
import { cn } from "@/lib/utils";

// Extracted Components
import { ConfidenceBadge } from "@/components/chat/confidence-badge";
import { CitationsPanel } from "@/components/chat/citations-panel";
import { ImageGallery } from "@/components/chat/image-gallery";
import { ProcessingBanner, type DocumentStatus } from "@/components/chat/processing-banner";

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

const QUICK_QUESTIONS = [
  "Summarize this document",
  "What are the key specifications?",
  "Show me the diagrams",
  "What are the safety warnings?",
  "List the main components",
  "What is the maintenance schedule?",
  "How do I troubleshoot common issues?",
  "What are the contact details?",
];

const BACKEND_URL = "http://127.0.0.1:8000";

export default function UserChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 1,
      role: "assistant",
      content:
        "Hi, I can answer questions from the uploaded documents using Hybrid Multimodal RAG. Select a document from the sidebar and ask about specifications, diagrams, or instructions — I'll cite my sources.",
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
    // FIX: Use h-[100dvh] so the page exactly fits the screen. No more scrolling down to see the input.
    <div className="flex h-[100dvh] flex-col bg-transparent">
      
      {/* Header */}
      <header className="shrink-0 border-b bg-white/90 px-4 py-3 backdrop-blur md:px-6">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4">
          <Link href="/" className="inline-flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-primary">
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Link>
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-primary-foreground shadow-sm">
              <BotMessageSquare className="h-5 w-5" />
            </div>
            <div className="hidden sm:block">
              <p className="text-sm font-semibold">Knowledge Assistant</p>
              <p className="text-xs text-muted-foreground">Hybrid Multimodal RAG</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Grid Area */}
      <main className="mx-auto flex w-full max-w-7xl flex-1 overflow-hidden px-4 py-4 md:px-6 md:py-6">
        <div className="grid w-full gap-6 lg:grid-cols-[340px_1fr] h-full">
          
          {/* ── Sidebar ─────────────────────────────────────────────────────── */}
          <aside className="hidden flex-col gap-4 overflow-y-auto pr-2 lg:flex">
            <section className="panel overflow-hidden shrink-0">
              <div className="bg-primary p-5 text-primary-foreground">
                <div className="flex items-center gap-3">
                  <FileText className="h-6 w-6" />
                  <div>
                    <h1 className="text-lg font-semibold">Explore Knowledge Base</h1>
                    <p className="mt-1 text-xs text-primary-foreground/80">Ask questions based on the uploaded manual.</p>
                  </div>
                </div>
              </div>
              <div className="space-y-4 p-5">
                <div>
                  <p className="text-sm text-muted-foreground">Active document</p>
                  <h2 className="mt-1 font-semibold line-clamp-1">
                    {activeDocument ? getProjectTitle(activeDocument) : "No document selected"}
                  </h2>
                  {activeDocument && (
                    <p className="mt-1 text-xs text-muted-foreground line-clamp-1">
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
                    className="field text-sm"
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
                    className="rounded text-primary focus:ring-primary"
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
              </div>
            </section>

            <section className="panel p-5 shrink-0">
              <h2 className="font-semibold text-sm">Quick questions</h2>
              <div className="mt-3 grid gap-2">
                {QUICK_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => sendMessage(q)}
                    className="rounded-md border bg-white px-3 py-2 text-left text-xs transition hover:border-primary/40 hover:bg-primary/5 disabled:opacity-50"
                    disabled={isLoading || !isDocReady}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </section>
          </aside>

          {/* ── Chat area ─────────────────────────────────────────────────────── */}
          <section className="panel flex h-full flex-col overflow-hidden bg-white shadow-sm border border-slate-200">
            <div className="shrink-0 border-b bg-slate-50/50 px-5 py-4">
              <h2 className="font-semibold text-slate-800">Chat with AI</h2>
              <p className="text-xs text-slate-500 mt-1">
                Answers come directly from retrieved document chunks. Sources are cited.
              </p>
            </div>

            {/* Processing warning banner */}
            {activeDocumentId && docStatus && docStatus.status !== "ready" && (
              <div className="shrink-0">
                <ProcessingBanner status={docStatus} documentId={activeDocumentId} />
              </div>
            )}

            {/* Scrollable messages area */}
            <div className="flex-1 overflow-y-auto p-4 md:p-5 space-y-6">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={cn("flex", msg.role === "user" ? "justify-end" : "justify-start")}
                >
                  <div
                    className={cn(
                      "max-w-[85%] sm:max-w-[75%] rounded-2xl px-5 py-4 text-sm shadow-sm",
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground rounded-tr-sm"
                        : msg.isProcessingWarning
                        ? "border border-amber-200 bg-amber-50 rounded-tl-sm"
                        : "border border-slate-200 bg-white text-slate-800 rounded-tl-sm",
                    )}
                  >
                    {msg.role === "assistant" && msg.confidence && (
                      <div className="mb-3">
                        <ConfidenceBadge level={msg.confidence} />
                      </div>
                    )}
                    <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                    <ImageGallery images={msg.images} imageResults={msg.imageResults} />
                    {msg.role === "assistant" && <CitationsPanel citations={msg.citations ?? []} />}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="inline-flex items-center gap-3 rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-5 py-4 text-sm text-slate-500 shadow-sm">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    Searching documents and generating grounded answer…
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Sticky Form at bottom */}
            <form onSubmit={handleSubmit} className="shrink-0 border-t bg-white p-4 sm:p-5">
              {!isDocReady && activeDocumentId && (
                <p className="mb-3 flex items-center gap-2 text-xs text-amber-600 font-medium bg-amber-50 rounded-md p-2">
                  <Clock className="h-4 w-4" />
                  Document is still indexing. Chat will be enabled once indexing completes.
                </p>
              )}
              <div className="flex gap-3">
                <input
                  className="field h-12 rounded-xl border-slate-300 bg-slate-50 text-sm shadow-sm transition-all focus:bg-white focus:ring-2 focus:ring-primary/20 focus:border-primary"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  disabled={!isDocReady}
                  placeholder={
                    !activeDocumentId
                      ? "Select a document from the sidebar to start chatting..."
                      : !isDocReady
                      ? `Indexing document (${docStatus?.progress ?? 0}%) — please wait…`
                      : "Ask about specifications, diagrams, maintenance, or troubleshooting…"
                  }
                />
                <button
                  type="submit"
                  className="inline-flex h-12 items-center gap-2 rounded-xl bg-primary px-6 text-sm font-semibold text-primary-foreground shadow-sm transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={isLoading || !isDocReady || input.trim().length === 0}
                >
                  <Send className="h-4 w-4" />
                  <span className="hidden sm:inline">Send</span>
                </button>
              </div>
            </form>

          </section>
        </div>
      </main>
    </div>
  );
}
