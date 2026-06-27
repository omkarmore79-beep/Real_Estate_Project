import { BuilderForm } from "@/components/builder-form";

export default function EditBuilderPage({ params }: { params: { id: string } }) {
  return <BuilderForm mode="edit" builderId={params.id} />;
}
