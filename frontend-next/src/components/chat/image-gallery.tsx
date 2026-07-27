import { resolveImageUrl, type ImageResult } from "@/lib/backend-data";

const BACKEND_URL = "http://127.0.0.1:8000";

export function ImageGallery({ images, imageResults }: { images?: string[]; imageResults?: ImageResult[] }) {
  type DisplayImage = { 
    url: string; 
    caption?: string; 
    page?: number | null; 
    type?: string;
    figure_number?: string;
    section?: string;
  };
  const items: DisplayImage[] = [];

  if (imageResults?.length) {
    for (const img of imageResults) {
      const url = resolveImageUrl(img, BACKEND_URL);
      if (url) {
        items.push({ 
          url, 
          caption: img.caption, 
          page: img.page_number, 
          type: img.image_type,
          figure_number: (img as any).figure_number,
          section: (img as any).section,
        });
      }
    }
  }
  if (images?.length && items.length === 0) {
    for (const img of images) {
      const url = resolveImageUrl(img, BACKEND_URL);
      if (url) items.push({ url });
    }
  }
  if (!items.length) return null;

  return (
    <div className="mt-3 grid gap-3 sm:grid-cols-2">
      {items.map((item, i) => (
        <div key={i} className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm transition hover:shadow">
          <img
            src={item.url}
            alt={item.caption || `Page image ${i + 1}`}
            className="max-h-72 w-full object-contain p-1"
            loading="lazy"
          />
          {(item.caption || item.type || item.figure_number || item.section) && (
            <div className="border-t bg-slate-50/80 px-3 py-2 text-xs border-slate-100">
              <div className="flex flex-wrap items-center justify-between gap-1.5 font-medium text-slate-700">
                <span className="font-semibold text-primary">{item.figure_number || item.type?.replace(/_/g, " ") || "Diagram"}</span>
                {item.page && <span className="text-slate-500">Page {item.page}</span>}
              </div>
              {item.section && (
                <p className="text-[10px] text-slate-500 uppercase tracking-wider mt-0.5">
                  Section: {item.section}
                </p>
              )}
              {item.caption && (
                <p className="mt-1 text-slate-500 line-clamp-2 leading-relaxed">{item.caption}</p>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
