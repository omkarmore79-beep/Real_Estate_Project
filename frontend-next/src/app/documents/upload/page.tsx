"use client";

import { ChangeEvent, FormEvent, useEffect, useState } from "react";
import Link from "next/link";
import { CheckCircle2, FileUp, Info, Loader2, Upload } from "lucide-react";
import { Badge, PageHeader } from "@/components/ui";

type UploadState = {
  status: "idle" | "uploading" | "success" | "error";
  message: string;
  documentId?: string;
  savedToMongo?: boolean;
};

export default function UploadDocumentPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [builderName, setBuilderName] = useState("");
  const [uploadState, setUploadState] = useState<UploadState>({
    status: "idle",
    message: "",
  });

  useEffect(() => {
    const builder = new URLSearchParams(window.location.search).get("builder");
    if (builder) {
      setBuilderName(builder);
    }
  }, []);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    setSelectedFile(event.target.files?.[0] ?? null);
    setUploadState({ status: "idle", message: "" });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedFile) {
      setUploadState({ status: "error", message: "Please choose a PDF brochure first." });
      return;
    }

    const form = event.currentTarget;
    const formData = new FormData(form);
    formData.set("file", selectedFile);

    setUploadState({ status: "uploading", message: "Processing brochure, JSON, and images..." });

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        setUploadState({ status: "error", message: data.error ?? "Upload failed." });
        return;
      }

      localStorage.setItem("activeDocumentId", data.document_id);
      setUploadState({
        status: "success",
        message: "Upload processed successfully. The chatbot will now answer from this file.",
        documentId: data.document_id,
        savedToMongo: data.saved_to_mongodb,
      });
    } catch {
      setUploadState({
        status: "error",
        message: "Something went wrong while uploading. Please try again.",
      });
    }
  }

  return (
    <>
      <PageHeader
        title="Upload document"
        description="Select the builder first, then upload one PDF per plan/project so chatbot answers stay scoped to that exact file."
      />
      <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
        <form onSubmit={handleSubmit} className="panel space-y-6 p-5">
          <div className="rounded-lg border border-dashed bg-secondary/60 p-8 text-center">
            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-md bg-white text-primary shadow-sm">
              <FileUp className="h-7 w-7" />
            </div>
            <p className="mt-4 font-semibold">{selectedFile ? selectedFile.name : "Choose a project brochure"}</p>
            <p className="mt-1 text-sm text-muted-foreground">PDF files work with the current extraction pipeline.</p>
            <label className="mt-4 inline-flex h-10 cursor-pointer items-center rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground">
              Browse files
              <input name="file" type="file" accept="application/pdf,.pdf" className="sr-only" onChange={handleFileChange} />
            </label>
          </div>

          <div className="grid gap-5 md:grid-cols-2">
            <label className="space-y-2">
              <span className="label">Builder name</span>
              <input
                name="builder"
                className="field"
                placeholder="Builder or developer name"
                required
                value={builderName}
                onChange={(event) => setBuilderName(event.target.value)}
              />
            </label>
            <label className="space-y-2">
              <span className="label">Project name</span>
              <input name="project" className="field" placeholder="Mahavir Park, Golden Palms..." required />
            </label>
            <label className="space-y-2">
              <span className="label">Type of document</span>
              <select name="document_type" className="field">
                <option value="">What does this document contain?</option>
                {["Brochure", "Floor plan", "Pricing", "RERA", "Amenities", "Location plan"].map((type) => (
                  <option key={type}>{type}</option>
                ))}
              </select>
            </label>
            <label className="space-y-2">
              <span className="label">Document title</span>
              <input name="title" className="field" placeholder="Project brochure or tower sheet" />
            </label>
            <label className="space-y-2 md:col-span-2">
              <span className="label">Short description</span>
              <textarea name="description" className="textarea-field" placeholder="Describe the important information this file contains." />
            </label>
            <label className="space-y-2">
              <span className="label">Tags / keywords</span>
              <input name="tags" className="field" placeholder="pricing, 2bhk, possession, amenities" />
            </label>
            <label className="space-y-2">
              <span className="label">Related tower / block</span>
              <input className="field" placeholder="Tower A, Jasmine East, Block 1" />
            </label>
            <label className="space-y-2">
              <span className="label">Related unit type</span>
              <input className="field" placeholder="2 BHK, 3 BHK, Penthouse" />
            </label>
            <label className="space-y-2">
              <span className="label">Upload date</span>
              <input className="field" type="date" defaultValue="2026-05-21" />
            </label>
            <label className="space-y-2 md:col-span-2">
              <span className="label">Notes</span>
              <textarea className="textarea-field" placeholder="Any indexing hints or document quality notes." />
            </label>
          </div>

          {uploadState.message ? (
            <div
              className={`rounded-md border px-4 py-3 text-sm ${
                uploadState.status === "error"
                  ? "border-red-200 bg-red-50 text-red-700"
                  : "border-primary/20 bg-primary/5 text-foreground"
              }`}
            >
              <div className="flex items-start gap-2">
                {uploadState.status === "uploading" ? <Loader2 className="mt-0.5 h-4 w-4 animate-spin" /> : null}
                {uploadState.status === "success" ? <CheckCircle2 className="mt-0.5 h-4 w-4 text-primary" /> : null}
                <div>
                  <p>{uploadState.message}</p>
                  {uploadState.documentId ? (
                    <p className="mt-1 text-xs text-muted-foreground">
                      Document ID: {uploadState.documentId} - MongoDB: {uploadState.savedToMongo ? "saved" : "not saved"}
                    </p>
                  ) : null}
                </div>
              </div>
            </div>
          ) : null}

          <div className="flex justify-end gap-3">
            {uploadState.status === "success" ? (
              <Link
                href={uploadState.documentId ? `/chat?documentId=${uploadState.documentId}` : "/chat"}
                className="inline-flex h-10 items-center rounded-md border bg-white px-4 text-sm font-semibold"
              >
                Ask chatbot
              </Link>
            ) : null}
            <button
              type="submit"
              className="inline-flex h-10 items-center gap-2 rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-60"
              disabled={uploadState.status === "uploading"}
            >
              {uploadState.status === "uploading" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
              Upload and process
            </button>
          </div>
        </form>

        <aside className="space-y-4">
          <div className="panel p-5">
            <div className="flex items-start gap-3">
              <Info className="mt-0.5 h-5 w-5 text-primary" />
              <div>
                <h2 className="font-semibold">Document type matters</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  After upload, the extracted JSON, raw text, PDF, and images are stored in MongoDB Atlas for hosted deployments.
                </p>
              </div>
            </div>
          </div>
          <div className="panel p-5">
            <h2 className="font-semibold">Suggested tags</h2>
            <div className="mt-3 flex flex-wrap gap-2">
              {["overview", "tower", "floor plan", "pricing", "legal", "rera", "amenities", "payment"].map((tag) => (
                <Badge key={tag}>{tag}</Badge>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </>
  );
}
