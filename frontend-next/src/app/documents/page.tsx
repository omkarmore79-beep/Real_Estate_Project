import { Upload } from "lucide-react";
import { DocumentTable } from "@/components/document-table";
import { ButtonLink, PageHeader } from "@/components/ui";
import { getUploadedProjects } from "@/lib/backend-data";

export default async function DocumentsPage() {
  const documents = await getUploadedProjects();

  return (
    <>
      <PageHeader
        title="Documents"
        description="Uploaded project files with document type and metadata for clean chatbot retrieval."
        action={
          <ButtonLink href="/documents/upload">
            <Upload className="h-4 w-4" />
            Upload document
          </ButtonLink>
        }
      />
      <div className="mb-4 grid gap-3 md:grid-cols-4">
        <input className="field md:col-span-2" placeholder="Search documents..." />
        <select className="field">
          <option>All document types</option>
          <option>Brochure</option>
          <option>Floor plan</option>
          <option>Pricing</option>
          <option>RERA</option>
        </select>
        <select className="field">
          <option>Newest first</option>
          <option>Oldest first</option>
        </select>
      </div>
      <DocumentTable documents={documents} />
    </>
  );
}
