import { Badge } from "@/components/ui";
import { DocumentActions } from "@/components/document-actions";
import {
  getBuilderName,
  getDocumentType,
  getProjectTitle,
  getUploadDate,
  type UploadedProject,
} from "@/lib/backend-data";

export function DocumentTable({ documents }: { documents: UploadedProject[] }) {
  return (
    <div className="panel overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[900px] text-sm">
          <thead className="bg-secondary text-left text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="px-5 py-3">Document</th>
              <th className="px-5 py-3">Builder</th>
              <th className="px-5 py-3">Project</th>
              <th className="px-5 py-3">Type</th>
              <th className="px-5 py-3">Uploaded</th>
              <th className="px-5 py-3">Images</th>
              <th className="px-5 py-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {documents.length > 0 ? (
              documents.map((document, index) => (
                <tr key={document.document_id ?? `${document.source_file ?? "document"}-${index}`} className="bg-white align-top">
                  <td className="px-5 py-4">
                    <p className="font-medium">{document.metadata?.title || getProjectTitle(document)}</p>
                    <p className="text-xs text-muted-foreground">{document.source_file}</p>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {(document.metadata?.tags ?? []).map((tag) => (
                        <Badge key={tag}>{tag}</Badge>
                      ))}
                    </div>
                  </td>
                  <td className="px-5 py-4">{getBuilderName(document)}</td>
                  <td className="px-5 py-4">{getProjectTitle(document)}</td>
                  <td className="px-5 py-4">
                    <Badge className="bg-primary/10 text-primary">{getDocumentType(document)}</Badge>
                  </td>
                  <td className="px-5 py-4">{getUploadDate(document)}</td>
                  <td className="px-5 py-4">{document.images?.length ?? 0}</td>
                  <td className="px-5 py-4">
                    <DocumentActions documentId={document.document_id} />
                  </td>
                </tr>
              ))
            ) : (
              <tr className="bg-white">
                <td colSpan={7} className="px-5 py-10 text-center text-sm text-muted-foreground">
                  No documents uploaded yet. Upload a PDF brochure to show it here.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
