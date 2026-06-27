import Link from "next/link";
import { FileSearch, Search } from "lucide-react";
import { Badge, PageHeader } from "@/components/ui";
import {
  getBuilderName,
  getDocumentType,
  getProjectTitle,
  getUploadDate,
  getUploadedProjects,
} from "@/lib/backend-data";

export default async function SearchDocumentsPage() {
  const documents = await getUploadedProjects();
  const builders = Array.from(new Set(documents.map(getBuilderName)));
  const projects = Array.from(new Set(documents.map(getProjectTitle)));
  const documentTypes = Array.from(new Set(documents.map(getDocumentType)));

  return (
    <>
      <PageHeader
        title="Search documents"
        description="Filter uploaded real estate documents by builder, project, document type, tags, and description."
      />
      <section className="panel mb-6 p-5">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input className="field pl-9" placeholder="Search: RERA, floor plan, payment plan..." />
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <select className="field">
            <option>All builders</option>
            {builders.map((builder) => (
              <option key={builder}>{builder}</option>
            ))}
          </select>
          <select className="field">
            <option>All projects</option>
            {projects.map((project) => (
              <option key={project}>{project}</option>
            ))}
          </select>
          <select className="field">
            <option>All document types</option>
            {documentTypes.map((type) => (
              <option key={type}>{type}</option>
            ))}
          </select>
        </div>
      </section>

      <section className="grid gap-4">
        {documents.length > 0 ? (
          documents.map((document, index) => (
            <article key={document.document_id ?? `${document.source_file ?? "document"}-${index}`} className="panel p-5">
              <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                <div className="flex gap-4">
                  <div className="hidden h-12 w-12 items-center justify-center rounded-md bg-primary/10 text-primary sm:flex">
                    <FileSearch className="h-6 w-6" />
                  </div>
                  <div>
                    <h2 className="font-semibold">{document.metadata?.title || getProjectTitle(document)}</h2>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {document.metadata?.description || "Processed brochure data is available for chatbot answers."}
                    </p>
                    <p className="mt-2 text-xs text-muted-foreground">
                      {getBuilderName(document)} -{" "}
                      <Link href="/chat" className="font-semibold text-primary">
                        {getProjectTitle(document)}
                      </Link>
                    </p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      <Badge className="bg-primary/10 text-primary">{getDocumentType(document)}</Badge>
                      {(document.metadata?.tags ?? []).map((tag) => (
                        <Badge key={tag}>{tag}</Badge>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="text-sm text-muted-foreground md:text-right">
                  <p>{document.source_file}</p>
                  <p>{document.images?.length ?? 0} images</p>
                  <p>{getUploadDate(document)}</p>
                </div>
              </div>
            </article>
          ))
        ) : (
          <div className="panel p-8 text-center text-sm text-muted-foreground">
            No uploaded documents found yet. Upload a PDF brochure to search real extracted data.
          </div>
        )}
      </section>
    </>
  );
}
