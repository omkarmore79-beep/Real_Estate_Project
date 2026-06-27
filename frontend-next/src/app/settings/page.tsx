import { PageHeader } from "@/components/ui";

export default function SettingsPage() {
  return (
    <>
      <PageHeader
        title="Settings"
        description="Placeholder settings for future chatbot indexing, file storage, and team permissions."
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <section className="panel p-5">
          <h2 className="font-semibold">Indexing preferences</h2>
          <div className="mt-5 space-y-4">
            <label className="flex items-center justify-between gap-4">
              <span>
                <span className="block text-sm font-medium">Auto-index uploaded documents</span>
                <span className="text-sm text-muted-foreground">Send new files to the chatbot retrieval pipeline.</span>
              </span>
              <input type="checkbox" className="h-5 w-5 accent-primary" defaultChecked />
            </label>
            <label className="space-y-2">
              <span className="label">Default document confidence threshold</span>
              <input type="range" min="0" max="100" defaultValue="78" className="w-full accent-primary" />
            </label>
          </div>
        </section>
        <section className="panel p-5">
          <h2 className="font-semibold">Storage configuration</h2>
          <div className="mt-5 space-y-4">
            <label className="space-y-2">
              <span className="label">Storage provider</span>
              <select className="field">
                <option>Local uploads</option>
                <option>S3 compatible bucket</option>
                <option>Vercel Blob</option>
              </select>
            </label>
            <label className="space-y-2">
              <span className="label">Webhook URL</span>
              <input className="field" placeholder="https://api.example.com/ingest" />
            </label>
          </div>
        </section>
      </div>
    </>
  );
}
