import Link from "next/link";
import { Edit, FileText, MapPin, Plus } from "lucide-react";
import { cookies } from "next/headers";
import { Badge, ButtonLink, PageHeader } from "@/components/ui";
import { getBuilderName, getProjectTitle, getUploadedProjects } from "@/lib/backend-data";

export default async function ProjectsPage() {
  const cookieStore = cookies();
  const domain = cookieStore.get("domain")?.value || "real-estate";
  const projects = await getUploadedProjects(domain);
  const isMachinery = domain === "machinery";

  return (
    <>
      <PageHeader
        title={isMachinery ? "Manuals" : "Projects"}
        description={isMachinery ? "Technical manuals and guides uploaded to the knowledge base." : "Project records created from uploaded real estate documents."}
        action={
          <ButtonLink href="/documents/upload">
            <Plus className="h-4 w-4" />
            Upload {isMachinery ? "manual" : "project"}
          </ButtonLink>
        }
      />
      <div className="grid gap-4 lg:grid-cols-2">
        {projects.length > 0 ? (
          projects.map((project, index) => (
            <article key={project.document_id ?? `${project.source_file ?? "project"}-${index}`} className="panel p-5">
              <div className="mb-4 flex items-start justify-between gap-4">
                <div>
                  <Link href={project.document_id ? `/chat?documentId=${project.document_id}` : "/chat"} className="text-lg font-semibold hover:text-primary">
                    {getProjectTitle(project)}
                  </Link>
                  <p className="mt-1 text-sm text-muted-foreground">{getBuilderName(project)}</p>
                </div>
                <ButtonLink href="/documents/upload" variant="secondary">
                  <Edit className="h-4 w-4" />
                </ButtonLink>
              </div>
              <div className="grid gap-3 text-sm sm:grid-cols-2">
                <p className="flex gap-2 text-muted-foreground">
                  <MapPin className="mt-0.5 h-4 w-4 shrink-0" />
                  {project.location || "Location not detected"}
                </p>
                <p className="font-mono text-xs text-muted-foreground">Source: {project.source_file || "Uploaded file"}</p>
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-2">
                <Badge>{project.property_type || "Project data"}</Badge>
                <Badge className="bg-primary/10 text-primary">
                  <FileText className="mr-1 h-3 w-3" />
                  1 document
                </Badge>
                {project.document_id ? (
                  <ButtonLink href={`/chat?documentId=${project.document_id}`} variant="secondary">
                    Ask this PDF
                  </ButtonLink>
                ) : null}
                {(project.buildings ?? []).map((tower) => (
                  <Badge key={tower}>{tower}</Badge>
                ))}
              </div>
            </article>
          ))
        ) : (
          <div className="panel p-8 text-center text-sm text-muted-foreground lg:col-span-2">
            No projects uploaded yet. Upload a PDF brochure to create the first project record.
          </div>
        )}
      </div>
    </>
  );
}
