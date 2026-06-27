import Link from "next/link";
import { FileText, Upload } from "lucide-react";
import { ButtonLink, PageHeader } from "@/components/ui";

export default function ProjectDetailPage() {
  return (
    <>
      <PageHeader
        title="Project details"
        description="Project detail pages are created from uploaded brochures. Open chat to ask questions from the active document."
        action={
          <ButtonLink href="/documents/upload">
            <Upload className="h-4 w-4" />
            Upload document
          </ButtonLink>
        }
      />
      <div className="panel p-8 text-center">
        <FileText className="mx-auto h-8 w-8 text-muted-foreground" />
        <p className="mt-3 font-semibold">No manual project detail selected</p>
        <Link href="/documents/upload" className="mt-1 inline-block text-sm font-semibold text-primary">
          Upload a real brochure
        </Link>
      </div>
    </>
  );
}
