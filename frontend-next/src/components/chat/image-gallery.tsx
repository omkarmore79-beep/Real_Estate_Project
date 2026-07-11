import { resolveImageUrl, type ImageResult } from "@/lib/backend-data";

const BACKEND_URL = "http://127.0.0.1:8000";

export function ImageGallery({ images, imageResults }: { images?: string[]; imageResults?: ImageResult[] }) {
  type DisplayImage = { url: string; caption?: string; page?: number | null; type?: string };
  const items: DisplayImage[] = [];

  if (imageResults?.length) {
    for (const img of imageResults) {
      const url = resolveImageUrl(img, BACKEND_URL);
      if (url) items.push({ url, caption: img.caption, page: img.page_number, type: img.image_type });
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
        <div key={i} className="overflow-hidden rounded-md border bg-white shadow-sm">
          <img
            src={item.url}
            alt={item.caption || `Page image ${i + 1}`}
            className="max-h-72 w-full object-contain"
            loading="lazy"
          />
          {(item.caption || item.type) && (
            <div className="border-t bg-secondary/50 px-2.5 py-1.5">
              {item.type && (
                <p className="text-xs font-medium capitalize text-primary">
                  {item.type.replace(/_/g, " ")}{item.page ? ` — p.${item.page}` : ""}
                </p>
              )}
              {item.caption && (
                <p className="line-clamp-2 text-xs text-muted-foreground">{item.caption}</p>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
