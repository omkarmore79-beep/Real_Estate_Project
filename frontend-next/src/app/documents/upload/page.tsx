"use client";

import { ChangeEvent, FormEvent, useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  AlertCircle,
  CheckCircle2,
  Database,
  FileText,
  FileUp,
  Info,
  Loader2,
  MessageSquare,
  RefreshCw,
  Upload,
} from "lucide-react";
import { Badge, PageHeader } from "@/components/ui";
import type { UploadResult } from "@/lib/backend-data";

// ── Types ──────────────────────────────────────────────────────────────────────
type ProcessingStatus =
  | "idle"
  | "uploading"
  | "uploaded"
  | "extracting_text"
  | "extracting_images"
  | "chunking"
  | "embedding_text"
  | "embedding_images"
  | "indexing_qdrant"
  | "ready"
  | "failed"
  | "error";

type UploadState = {
  phase: "idle" | "uploading" | "processing" | "done" | "error";
  status: ProcessingStatus;
  message: string;
  progress: number;
  documentId?: string;
  filename?: string;
  result?: UploadResult & {
    text_chunks_indexed?: number;
    images_indexed?: number;
    total_pages?: number;
    qdrant_status?: string;
    ocr_used?: boolean;
    saved_to_mongodb?: boolean;
  };
  statusResult?: {
    text_chunks_indexed: number;
    images_indexed: number;
    total_pages: number;
    error?: string;
  };
};

// ── Step metadata ──────────────────────────────────────────────────────────────
const PIPELINE_STEPS: { status: ProcessingStatus; label: string }[] = [
  { status: "uploaded", label: "File received" },
  { status: "extracting_text", label: "Extracting text" },
  { status: "extracting_images", label: "Extracting images" },
  { status: "running_ocr", label: "Running OCR scans" },
  { status: "chunking", label: "Chunking text" },
  { status: "embedding_text", label: "Text embeddings (bge-m3)" },
  { status: "embedding_images", label: "Image embeddings (jina-clip-v2)" },
  { status: "indexing_qdrant", label: "Indexing into Qdrant" },
  { status: "ready", label: "Ready for chat" },
];


const STATUS_ORDER = PIPELINE_STEPS.map((s) => s.status);

function stepIndex(status: ProcessingStatus) {
  const idx = STATUS_ORDER.indexOf(status);
  return idx === -1 ? -1 : idx;
}

// ── Polling hook ───────────────────────────────────────────────────────────────
function useStatusPoller(
  documentId: string | undefined,
  active: boolean,
  onUpdate: (data: {
    status: ProcessingStatus;
    progress: number;
    message: string;
    text_chunks_indexed: number;
    images_indexed: number;
    total_pages: number;
    error?: string;
  }) => void,
) {
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!active || !documentId) {
      stop();
      return;
    }

    const poll = async () => {
      try {
        const res = await fetch(`/api/documents/${documentId}/status`, {
          cache: "no-store",
        });
        if (!res.ok) return;
        const data = await res.json();
        onUpdate(data);
        if (data.status === "ready" || data.status === "failed") {
          stop();
        }
      } catch {
        // silently ignore network errors — will retry next interval
      }
    };

    poll(); // immediate first check
    intervalRef.current = setInterval(poll, 3000);
    return stop;
  }, [documentId, active, onUpdate, stop]);
}

// ── Main component ─────────────────────────────────────────────────────────────
export default function UploadDocumentPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [builderName, setBuilderName] = useState("");
  const [uploadState, setUploadState] = useState<UploadState>({
    phase: "idle",
    status: "idle",
    message: "",
    progress: 0,
  });

  useEffect(() => {
    const builder = new URLSearchParams(window.location.search).get("builder");
    if (builder) setBuilderName(builder);

    // Resume polling if activeDocumentId exists in localStorage
    const savedDocId = localStorage.getItem("activeDocumentId");
    if (savedDocId) {
      fetch(`/api/documents/${savedDocId}/status`, { cache: "no-store" })
        .then((res) => res.json())
        .then((data) => {
          if (data && data.status) {
            setUploadState({
              phase: data.status === "ready" ? "done" : data.status === "failed" ? "error" : "processing",
              status: data.status,
              message: data.message || "",
              progress: data.progress ?? (data.status === "ready" ? 100 : 0),
              documentId: savedDocId,
              statusResult: {
                text_chunks_indexed: data.text_chunks_indexed || 0,
                images_indexed: data.images_indexed || 0,
                total_pages: data.total_pages || 0,
                error: data.error,
              }
            });
          }
        })
        .catch(() => {});
    }
  }, []);

  const isPolling =
    uploadState.phase === "processing" &&
    uploadState.documentId != null;

  const handleStatusUpdate = useCallback(
    (data: {
      status: ProcessingStatus;
      progress: number;
      message: string;
      text_chunks_indexed: number;
      images_indexed: number;
      total_pages: number;
      error?: string;
    }) => {
      setUploadState((prev) => ({
        ...prev,
        status: data.status,
        progress: data.progress ?? prev.progress,
        message: data.message || prev.message,
        phase:
          data.status === "ready"
            ? "done"
            : data.status === "failed"
            ? "error"
            : "processing",
        statusResult: {
          text_chunks_indexed: data.text_chunks_indexed,
          images_indexed: data.images_indexed,
          total_pages: data.total_pages,
          error: data.error,
        },
      }));
    },
    [],
  );


  useStatusPoller(uploadState.documentId, isPolling, handleStatusUpdate);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    setSelectedFile(event.target.files?.[0] ?? null);
    setUploadState({ phase: "idle", status: "idle", message: "", progress: 0 });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedFile) {
      setUploadState({
        phase: "error",
        status: "error",
        message: "Please choose a PDF brochure first.",
        progress: 0,
      });
      return;
    }

    const form = event.currentTarget;
    const formData = new FormData(form);
    formData.set("file", selectedFile);

    setUploadState({
      phase: "uploading",
      status: "uploading",
      message: "Uploading PDF to server…",
      progress: 2,
    });

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        setUploadState({
          phase: "error",
          status: "error",
          message: data.error ?? data.detail ?? "Upload failed.",
          progress: 0,
        });
        return;
      }

      // Upload successful — backend returns immediately, indexing is in background
      if (data.document_id) {
        localStorage.setItem("activeDocumentId", data.document_id);
      }
      setUploadState({
        phase: "processing",
        status: "uploaded",
        message:
          "Document uploaded successfully. Multimodal hybrid RAG indexing started.",
        progress: 5,
        documentId: data.document_id,
        filename: data.filename,
        result: data,
      });

    } catch {
      setUploadState({
        phase: "error",
        status: "error",
        message: "Upload failed. Is the backend running?",
        progress: 0,
      });
    }
  }

  const currentStepIdx = stepIndex(uploadState.status as ProcessingStatus);
  const sr = uploadState.statusResult;
  const isReady = uploadState.status === "ready";
  const isFailed =
    uploadState.status === "failed" || uploadState.phase === "error";
  const isActive =
    uploadState.phase === "uploading" || uploadState.phase === "processing";

  return (
    <>
      <PageHeader
        title="Upload document"
        description="Upload a real estate PDF — the backend extracts text and images, creates embeddings, and indexes everything into Qdrant automatically."
      />
      <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
        {/* ── Form ────────────────────────────────────────────────────── */}
        <form onSubmit={handleSubmit} className="panel space-y-6 p-5">
          {/* File drop zone */}
          <div className="rounded-lg border border-dashed bg-secondary/60 p-8 text-center">
            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-md bg-white text-primary shadow-sm">
              <FileUp className="h-7 w-7" />
            </div>
            <p className="mt-4 font-semibold">
              {selectedFile ? selectedFile.name : "Choose a project brochure (PDF)"}
            </p>
            {selectedFile && (
              <p className="mt-1 text-xs text-muted-foreground">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            )}
            <p className="mt-1 text-sm text-muted-foreground">
              Text and images are extracted directly from the PDF layer. No OCR.
            </p>
            <label className="mt-4 inline-flex h-10 cursor-pointer items-center rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground">
              Browse files
              <input
                name="file"
                type="file"
                accept="application/pdf,.pdf"
                className="sr-only"
                onChange={handleFileChange}
                disabled={isActive}
              />
            </label>
          </div>

          {/* Metadata fields */}
          <div className="grid gap-5 md:grid-cols-2">
            <label className="space-y-2">
              <span className="label">Builder name</span>
              <input
                name="builder"
                className="field"
                placeholder="Builder or developer name"
                required
                value={builderName}
                onChange={(e) => setBuilderName(e.target.value)}
                disabled={isActive}
              />
            </label>
            <label className="space-y-2">
              <span className="label">Project name</span>
              <input
                name="project"
                className="field"
                placeholder="Mahavir Park, Golden Palms…"
                required
                disabled={isActive}
              />
            </label>
            <label className="space-y-2">
              <span className="label">Document type</span>
              <select name="document_type" className="field" disabled={isActive}>
                <option value="">What does this document contain?</option>
                {["Brochure", "Floor plan", "Pricing", "RERA", "Amenities", "Location plan"].map(
                  (t) => <option key={t}>{t}</option>,
                )}
              </select>
            </label>
            <label className="space-y-2">
              <span className="label">Document title</span>
              <input
                name="title"
                className="field"
                placeholder="Project brochure or tower sheet"
                disabled={isActive}
              />
            </label>
            <label className="space-y-2 md:col-span-2">
              <span className="label">Short description</span>
              <textarea
                name="description"
                className="textarea-field"
                placeholder="Describe the important information this file contains."
                disabled={isActive}
              />
            </label>
            <label className="space-y-2">
              <span className="label">Tags / keywords</span>
              <input
                name="tags"
                className="field"
                placeholder="pricing, 2bhk, possession, amenities"
                disabled={isActive}
              />
            </label>
          </div>

          {/* ── Status panel ──────────────────────────────────────────── */}
          {uploadState.phase !== "idle" && (
            <div
              className={`rounded-lg border p-4 ${
                isFailed
                  ? "border-red-200 bg-red-50"
                  : isReady
                  ? "border-emerald-200 bg-emerald-50"
                  : "border-primary/20 bg-primary/5"
              }`}
            >
              {/* Header */}
              <div className="flex items-start gap-2">
                {isFailed ? (
                  <AlertCircle className="mt-0.5 h-4 w-4 text-red-500 shrink-0" />
                ) : isReady ? (
                  <CheckCircle2 className="mt-0.5 h-4 w-4 text-emerald-600 shrink-0" />
                ) : (
                  <Loader2 className="mt-0.5 h-4 w-4 animate-spin text-primary shrink-0" />
                )}
                <p className="text-sm font-medium text-foreground">
                  {uploadState.message}
                </p>
              </div>

              {/* Progress bar */}
              {!isFailed && (
                <div className="mt-3">
                  <div className="mb-1 flex justify-between text-xs text-muted-foreground">
                    <span>{uploadState.status.replace(/_/g, " ")}</span>
                    <span>{uploadState.progress}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-white/60">
                    <div
                      className="h-full rounded-full bg-primary transition-all duration-500"
                      style={{ width: `${uploadState.progress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Pipeline step tracker */}
              {uploadState.phase !== "uploading" && (
                <div className="mt-4 grid grid-cols-4 gap-2 sm:grid-cols-8">
                  {PIPELINE_STEPS.map((step, i) => {
                    const done =
                      isReady || (currentStepIdx >= 0 && i < currentStepIdx);
                    const active = i === currentStepIdx;
                    return (
                      <div key={step.status} className="flex flex-col items-center gap-1">
                        <div
                          className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-semibold transition-colors ${
                            isFailed && active
                              ? "bg-red-500 text-white"
                              : done || (isReady && i < PIPELINE_STEPS.length)
                              ? "bg-emerald-500 text-white"
                              : active
                              ? "bg-primary text-primary-foreground"
                              : "bg-white text-muted-foreground ring-1 ring-border"
                          }`}
                        >
                          {done || isReady ? (
                            <CheckCircle2 className="h-3.5 w-3.5" />
                          ) : (
                            i + 1
                          )}
                        </div>
                        <p className="text-center text-[9px] leading-tight text-muted-foreground">
                          {step.label}
                        </p>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Stats on completion */}
              {isReady && sr && (
                <div className="mt-4 grid grid-cols-3 gap-2">
                  {[
                    { label: "Pages", value: sr.total_pages },
                    { label: "Text chunks", value: sr.text_chunks_indexed },
                    { label: "Images", value: sr.images_indexed },
                  ].map(({ label, value }) => (
                    <div
                      key={label}
                      className="rounded-md border border-emerald-200 bg-white px-2 py-2 text-center"
                    >
                      <p className="text-lg font-bold text-emerald-700">{value ?? "—"}</p>
                      <p className="text-xs text-muted-foreground">{label} indexed</p>
                    </div>
                  ))}
                </div>
              )}

              {/* Metadata row */}
              {uploadState.result?.document_id && (
                <div className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
                  <span>ID: {uploadState.result.document_id}</span>
                  <span>·</span>
                  <span className="flex items-center gap-1">
                    <Database className="h-3 w-3" />
                    MongoDB:{" "}
                    {uploadState.result.saved_to_mongodb ? "saved" : "not saved"}
                  </span>
                  <span>·</span>
                  <span>OCR: {uploadState.result.ocr_used ? "yes" : "no"}</span>
                </div>
              )}

              {/* Error detail */}
              {isFailed && sr?.error && (
                <p className="mt-2 rounded bg-red-100 px-2 py-1 text-xs text-red-700">
                  Error: {sr.error}
                </p>
              )}
            </div>
          )}

          {/* Action buttons */}
          <div className="flex flex-wrap justify-end gap-3">
            {isReady && uploadState.documentId && (
              <Link
                href={`/chat?documentId=${uploadState.documentId}`}
                className="inline-flex h-10 items-center gap-2 rounded-md bg-emerald-600 px-4 text-sm font-semibold text-white hover:bg-emerald-700"
              >
                <MessageSquare className="h-4 w-4" />
                Start chatting
              </Link>
            )}
            {isFailed && (
              <button
                type="button"
                onClick={() =>
                  setUploadState({
                    phase: "idle",
                    status: "idle",
                    message: "",
                    progress: 0,
                  })
                }
                className="inline-flex h-10 items-center gap-2 rounded-md border px-4 text-sm font-semibold"
              >
                <RefreshCw className="h-4 w-4" />
                Try again
              </button>
            )}
            <button
              type="submit"
              className="inline-flex h-10 items-center gap-2 rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-60"
              disabled={isActive || isReady}
            >
              {uploadState.phase === "uploading" ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Upload className="h-4 w-4" />
              )}
              {uploadState.phase === "uploading" ? "Uploading…" : "Upload and index"}
            </button>
          </div>
        </form>

        {/* ── Sidebar ─────────────────────────────────────────────────── */}
        <aside className="space-y-4">
          <div className="panel p-5">
            <div className="flex items-start gap-3">
              <Info className="mt-0.5 h-5 w-5 text-primary shrink-0" />
              <div>
                <h2 className="font-semibold">What happens on upload?</h2>
                <ul className="mt-2 space-y-1.5 text-sm text-muted-foreground">
                  <li className="flex gap-2">
                    <FileText className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                    Text extracted per page (PyMuPDF, no OCR)
                  </li>
                  <li className="flex gap-2">
                    <FileText className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                    Each page rendered as PNG image
                  </li>
                  <li className="flex gap-2">
                    <Database className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                    Text chunks embedded with bge-m3
                  </li>
                  <li className="flex gap-2">
                    <Database className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                    Images embedded with jina-clip-v2
                  </li>
                  <li className="flex gap-2">
                    <Database className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                    Vectors stored in Qdrant
                  </li>
                  <li className="flex gap-2">
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                    Metadata saved to MongoDB
                  </li>
                </ul>
                <p className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800">
                  Embedding models download ~5 GB on first run. Subsequent runs
                  use cached models and are much faster.
                </p>
              </div>
            </div>
          </div>

          <div className="panel p-5">
            <h2 className="font-semibold">Suggested tags</h2>
            <div className="mt-3 flex flex-wrap gap-2">
              {[
                "overview", "tower", "floor plan", "pricing",
                "legal", "rera", "amenities", "payment",
              ].map((tag) => (
                <Badge key={tag}>{tag}</Badge>
              ))}
            </div>
          </div>

          <div className="panel p-5">
            <h2 className="font-semibold">Processing pipeline</h2>
            <div className="mt-3 space-y-1.5">
              {PIPELINE_STEPS.map((step, i) => {
                const done =
                  isReady ||
                  (currentStepIdx >= 0 && i < currentStepIdx);
                const active = i === currentStepIdx && !isReady && !isFailed;
                return (
                  <div key={step.status} className="flex items-center gap-2 text-sm">
                    <div
                      className={`h-2 w-2 rounded-full shrink-0 ${
                        done
                          ? "bg-emerald-500"
                          : active
                          ? "bg-primary animate-pulse"
                          : "bg-border"
                      }`}
                    />
                    <span
                      className={
                        done
                          ? "text-emerald-700"
                          : active
                          ? "font-medium text-foreground"
                          : "text-muted-foreground"
                      }
                    >
                      {step.label}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </aside>
      </div>
    </>
  );
}
