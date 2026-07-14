import Link from "next/link";
import { Building2, Eye, FileText, FolderKanban, Search } from "lucide-react";
import { cookies } from "next/headers";
import { Badge, ButtonLink, PageHeader, StatCard } from "@/components/ui";
import { getUploadedProjects, type UploadedProject } from "@/lib/backend-data";

export default async function DashboardPage() {
  const cookieStore = cookies();
  const domain = cookieStore.get("domain")?.value || "real-estate";
  const uploadedProjects = await getUploadedProjects(domain);
  const recentDocuments = uploadedProjects.slice(0, 4);
  const builderCount = new Set(
    uploadedProjects.map((project) => project.developer || project.metadata?.builder).filter(Boolean),
  ).size;
  const isMachinery = domain === "machinery";

  return (
    <>
      <PageHeader
        title="Dashboard"
        description="Manage real uploaded project brochures before chatbot search."
        action={<ButtonLink href="/documents/upload">Upload document</ButtonLink>}
      />

      <section className="grid gap-4 md:grid-cols-3">
        <StatCard label="Total builders" value={builderCount} icon={Building2} />
        <StatCard label="Total projects" value={uploadedProjects.length} icon={FolderKanban} tone="amber" />
        <StatCard label="Total documents" value={uploadedProjects.length} icon={FileText} tone="slate" />
      </section>

      <section className="mt-6 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="panel overflow-hidden">
          <div className="flex items-center justify-between border-b px-5 py-4">
            <div>
              <h2 className="font-semibold">Recently uploaded documents</h2>
              <p className="text-sm text-muted-foreground">Real files processed by the extraction pipeline.</p>
            </div>
            <Link href="/documents/upload" className="text-sm font-semibold text-primary">
              Upload new
            </Link>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-sm">
              <thead className="bg-secondary text-left text-xs uppercase tracking-wide text-muted-foreground">
                <tr>
                  <th className="px-5 py-3">Document</th>
                  <th className="px-5 py-3">Project</th>
                  <th className="px-5 py-3">Type</th>
                  <th className="px-5 py-3">Uploaded</th>
                  <th className="px-5 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {recentDocuments.length > 0 ? (
                  recentDocuments.map((document, index) => (
                    <tr key={document.document_id ?? `${document.source_file ?? "document"}-${index}`} className="bg-white">
                      <td className="px-5 py-4">
                        <p className="font-medium">{document.metadata?.title || document.project_name || "Uploaded brochure"}</p>
                        <p className="text-xs text-muted-foreground">{document.source_file}</p>
                      </td>
                      <td className="px-5 py-4">{document.project_name || document.metadata?.builder || "Detected after processing"}</td>
                      <td className="px-5 py-4">
                        <Badge>{document.metadata?.document_type || document.property_type || "Brochure"}</Badge>
                      </td>
                      <td className="px-5 py-4">
                        {document.uploaded_at ? new Date(document.uploaded_at).toLocaleDateString() : "Recent"}
                      </td>
                      <td className="px-5 py-4">
                        <div className="flex justify-end gap-2">
                          <Link
                            href={document.document_id ? `/chat?documentId=${document.document_id}` : "/chat"}
                            className="rounded-md border p-2 text-muted-foreground transition hover:bg-secondary hover:text-foreground"
                            aria-label="Open chatbot"
                          >
                            <Eye className="h-4 w-4" />
                          </Link>
                        </div>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr className="bg-white">
                    <td colSpan={5} className="px-5 py-10 text-center text-sm text-muted-foreground">
                      No real brochures uploaded yet. Upload a PDF to populate this dashboard.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="panel p-5">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h2 className="font-semibold">Quick search</h2>
              <p className="text-sm text-muted-foreground">Ask from extracted real brochure data.</p>
            </div>
            <Search className="h-5 w-5 text-muted-foreground" />
          </div>
          <input className="field" placeholder="Try: floor plan, RERA, pricing..." />
          <div className="mt-4 space-y-3">
            {recentDocuments.length > 0 ? (
              recentDocuments.slice(0, 3).map((document, index) => (
                <Link
                  href={document.document_id ? `/chat?documentId=${document.document_id}` : "/chat"}
                  key={document.document_id ?? `${document.source_file ?? "document"}-${index}`}
                  className="block rounded-md border bg-white p-3 transition hover:border-primary/40 hover:bg-primary/5"
                >
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm font-medium">{document.project_name || document.metadata?.title || "Uploaded brochure"}</p>
                    <Badge>{document.metadata?.document_type || "Brochure"}</Badge>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">{document.source_file}</p>
                </Link>
              ))
            ) : (
              <p className="rounded-md border bg-white p-3 text-sm text-muted-foreground">
                Search will use uploaded brochure data after the first successful upload.
              </p>
            )}
          </div>
        </div>
      </section>
    </>
  );
}
