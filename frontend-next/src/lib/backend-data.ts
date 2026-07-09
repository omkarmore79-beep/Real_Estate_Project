export type UploadedProject = {
  document_id?: string;
  project_name?: string;
  buildings?: string[];
  developer?: string;
  location?: string;
  property_type?: string;
  source_file?: string;
  uploaded_at?: string;
  images?: unknown[];
  metadata?: {
    title?: string;
    builder?: string;
    project?: string;
    document_type?: string;
    description?: string;
    tags?: string[];
  };
};

export type BuilderGroup = {
  name: string;
  documents: UploadedProject[];
  project_count: number;
};

// ── RAG Response Types ────────────────────────────────────────────────────────

export type Citation = {
  citation_id?: string;
  document_id?: string;
  source_file?: string;
  page_number?: number | null;
  source_type?: string;
  section?: string;
  snippet?: string;
  ocr_used?: boolean;
};

export type ImageResult = {
  document_id?: string;
  image_id?: string;
  image_url?: string;
  page_number?: number | null;
  image_type?: string;
  caption?: string;
};

export type RAGChatResponse = {
  question: string;
  answer: string;
  citations?: Citation[];
  images?: (string | ImageResult)[];
  confidence?: "high" | "medium" | "low";
  intent?: {
    intent?: string;
    requires_visual_response?: boolean;
    requires_image?: boolean;
    requires_text?: boolean;
    image_types?: string[];
    detected_image_types?: string[];
  };
  retrieved_context?: unknown[];
};

export type UploadResult = {
  // v3: returned immediately — indexing happens in background
  document_id?: string;
  status?: "processing" | "ready" | "failed";
  filename?: string;
  message?: string;
  saved_to_mongodb?: boolean;
  ocr_used?: boolean;
  // legacy fields (v2 sync response)
  total_pages?: number;
  text_chunks_indexed?: number;
  images_indexed?: number;
  qdrant_status?: string;
  data?: unknown;
};

export type DocumentStatus = {
  document_id?: string;
  status:
    | "uploaded"
    | "extracting_text"
    | "extracting_images"
    | "running_ocr"
    | "chunking"
    | "embedding_text"
    | "embedding_images"
    | "indexing_qdrant"
    | "ready"
    | "failed"
    | string;
  progress: number;
  message: string;
  text_chunks_indexed?: number;
  images_indexed?: number;
  total_pages?: number;
  error?: string;
};


// ── API helpers ────────────────────────────────────────────────────────────────

export const backendBaseUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function getUploadedProjects(): Promise<UploadedProject[]> {
  try {
    const response = await fetch(`${backendBaseUrl}/projects`, { cache: "no-store" });
    if (!response.ok) return [];
    const data = await response.json();
    return Array.isArray(data.projects) ? data.projects : [];
  } catch {
    return [];
  }
}

export async function getBuilderGroups(): Promise<BuilderGroup[]> {
  try {
    const response = await fetch(`${backendBaseUrl}/builders`, { cache: "no-store" });
    if (!response.ok) return [];
    const data = await response.json();
    return Array.isArray(data.builders) ? data.builders : [];
  } catch {
    return [];
  }
}

export function getProjectTitle(project: UploadedProject) {
  return project.project_name || project.metadata?.project || project.metadata?.title || "Uploaded brochure";
}

export function getBuilderName(project: UploadedProject) {
  return project.developer || project.metadata?.builder || "Builder not detected";
}

export function getDocumentType(project: UploadedProject) {
  return project.metadata?.document_type || project.property_type || "Brochure";
}

export function getUploadDate(project: UploadedProject) {
  return project.uploaded_at ? new Date(project.uploaded_at).toLocaleDateString() : "Recent";
}

export function getPdfUrl(project: UploadedProject) {
  if (!project.document_id) return "";
  return `${backendBaseUrl}/documents/${project.document_id}/pdf`;
}

/** Resolve an image reference (string path or ImageResult object) to a usable URL. */
export function resolveImageUrl(
  image: string | ImageResult,
  backendUrl: string = "http://127.0.0.1:8000",
): string {
  if (typeof image === "string") {
    if (image.startsWith("http") || image.startsWith("/")) return image;
    return `${backendUrl}/${image}`;
  }
  // ImageResult object
  const url = image.image_url;
  if (!url) return "";
  if (url.startsWith("http") || url.startsWith("data:")) return url;
  return `${backendUrl}${url.startsWith("/") ? "" : "/"}${url}`;
}
