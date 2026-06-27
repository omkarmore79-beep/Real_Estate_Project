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
