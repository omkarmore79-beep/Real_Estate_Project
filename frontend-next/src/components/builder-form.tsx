import { PageHeader } from "@/components/ui";

export function BuilderForm({ mode }: { mode: "add" | "edit"; builderId?: string }) {
  return (
    <>
      <PageHeader
        title={mode === "add" ? "Add builder" : "Edit builder"}
        description="Manual builder editing is not connected yet. Builder names are detected from uploaded brochures."
      />
      <form className="panel max-w-3xl space-y-5 p-5">
        <div className="grid gap-5 md:grid-cols-2">
          <label className="space-y-2">
            <span className="label">Builder name</span>
            <input className="field" placeholder="Builder or developer name" />
          </label>
          <label className="space-y-2">
            <span className="label">Contact person</span>
            <input className="field" placeholder="Sales manager name" />
          </label>
          <label className="space-y-2">
            <span className="label">Phone</span>
            <input className="field" placeholder="+91 98765 43210" />
          </label>
          <label className="space-y-2">
            <span className="label">Email</span>
            <input className="field" placeholder="contact@builder.com" />
          </label>
        </div>
        <label className="space-y-2">
          <span className="label">Builder address</span>
          <textarea className="textarea-field" placeholder="Registered office address" />
        </label>
        <div className="flex justify-end gap-3">
          <button type="button" className="h-10 rounded-md border bg-white px-4 text-sm font-semibold">
            Cancel
          </button>
          <button type="submit" className="h-10 rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground">
            Save builder
          </button>
        </div>
      </form>
    </>
  );
}
