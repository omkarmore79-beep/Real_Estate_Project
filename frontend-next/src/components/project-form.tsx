import { PageHeader } from "@/components/ui";

export function ProjectForm({ mode }: { mode: "add" | "edit"; projectId?: string }) {
  return (
    <>
      <PageHeader
        title={mode === "add" ? "Add project" : "Edit project"}
        description="Manual project editing is not connected yet. Uploaded brochures create project records automatically."
      />
      <form className="panel max-w-4xl space-y-5 p-5">
        <div className="grid gap-5 md:grid-cols-2">
          <label className="space-y-2">
            <span className="label">Project name</span>
            <input className="field" placeholder="Project name" />
          </label>
          <label className="space-y-2">
            <span className="label">Builder</span>
            <input className="field" placeholder="Builder or developer name" />
          </label>
          <label className="space-y-2 md:col-span-2">
            <span className="label">Project location/address</span>
            <input className="field" placeholder="Street, sector, landmark" />
          </label>
          <label className="space-y-2">
            <span className="label">City</span>
            <input className="field" placeholder="City" />
          </label>
          <label className="space-y-2">
            <span className="label">State</span>
            <input className="field" placeholder="State" />
          </label>
          <label className="space-y-2">
            <span className="label">RERA number</span>
            <input className="field" placeholder="RERA registration number" />
          </label>
          <label className="space-y-2">
            <span className="label">Project status</span>
            <select className="field" defaultValue="Under construction">
              <option>Planning</option>
              <option>Under construction</option>
              <option>Ready to move</option>
              <option>Delivered</option>
            </select>
          </label>
        </div>
        <label className="space-y-2">
          <span className="label">Towers / blocks</span>
          <input className="field" placeholder="Tower A, Tower B, Block 1" />
        </label>
        <div className="flex justify-end gap-3">
          <button type="button" className="h-10 rounded-md border bg-white px-4 text-sm font-semibold">
            Cancel
          </button>
          <button type="submit" className="h-10 rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground">
            Save project
          </button>
        </div>
      </form>
    </>
  );
}
