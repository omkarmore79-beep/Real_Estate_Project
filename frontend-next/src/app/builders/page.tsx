import { Plus, Upload } from "lucide-react";
import { DocumentActions } from "@/components/document-actions";
import { Badge, ButtonLink, PageHeader } from "@/components/ui";
import { getBuilderGroups, getDocumentType, getProjectTitle, getUploadDate } from "@/lib/backend-data";

export default async function BuildersPage() {
  const builders = await getBuilderGroups();

  return (
    <>
      <PageHeader
        title="Builders"
        description="Builder details detected from uploaded project documents."
        action={
          <ButtonLink href="/documents/upload">
            <Plus className="h-4 w-4" />
            Upload document
          </ButtonLink>
        }
      />
      <div className="grid gap-4">
        {builders.length > 0 ? (
          builders.map((builder) => (
            <article key={builder.name} className="panel p-5">
              <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h2 className="text-lg font-semibold">{builder.name}</h2>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <Badge>{builder.project_count} PDFs</Badge>
                    <Badge>{builder.project_count} projects/plans</Badge>
                  </div>
                </div>
                <ButtonLink href={`/documents/upload?builder=${encodeURIComponent(builder.name)}`} variant="secondary">
                  <Upload className="h-4 w-4" />
                  Upload PDF
                </ButtonLink>
              </div>
              <div className="overflow-hidden rounded-md border">
                <table className="w-full min-w-[760px] text-sm">
                  <thead className="bg-secondary text-left text-xs uppercase tracking-wide text-muted-foreground">
                    <tr>
                      <th className="px-4 py-3">Project / PDF</th>
                      <th className="px-4 py-3">Type</th>
                      <th className="px-4 py-3">Location</th>
                      <th className="px-4 py-3">Uploaded</th>
                      <th className="px-4 py-3 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y bg-white">
                    {builder.documents.map((document, index) => (
                      <tr key={document.document_id ?? `${document.source_file ?? "document"}-${index}`}>
                        <td className="px-4 py-3">
                          <p className="font-medium">{getProjectTitle(document)}</p>
                          <p className="text-xs text-muted-foreground">{document.source_file || "Uploaded PDF"}</p>
                        </td>
                        <td className="px-4 py-3">
                          <Badge className="bg-primary/10 text-primary">{getDocumentType(document)}</Badge>
                        </td>
                        <td className="px-4 py-3 text-muted-foreground">{document.location || "Location not detected"}</td>
                        <td className="px-4 py-3">{getUploadDate(document)}</td>
                        <td className="px-4 py-3">
                          <DocumentActions documentId={document.document_id} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          ))
        ) : (
          <div className="panel p-8 text-center text-sm text-muted-foreground">
            No builders found yet. They will appear after you upload and process a real brochure.
          </div>
        )}
      </div>
    </>
  );
}
