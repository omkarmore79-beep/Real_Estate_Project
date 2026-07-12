import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import type { Citation, ImageResult } from "@/lib/backend-data";

type BackendChatResponse = {
  question?: string;
  answer?: string;
  // Legacy: plain image paths
  images?: (string | ImageResult)[];
  // RAG fields
  citations?: Citation[];
  confidence?: "high" | "medium" | "low";
  intent?: Record<string, unknown>;
  retrieved_context?: unknown[];
};

const backendBaseUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

function resolveImage(image: string | ImageResult): string | ImageResult {
  if (typeof image === "string") {
    if (image.startsWith("http://") || image.startsWith("https://")) return image;
    return `${backendBaseUrl}/${image.replace(/^\/+/, "")}`;
  }
  // ImageResult object — resolve image_url if it's a relative path
  if (image.image_url && !image.image_url.startsWith("http") && !image.image_url.startsWith("data:")) {
    return {
      ...image,
      image_url: `${backendBaseUrl}${image.image_url.startsWith("/") ? "" : "/"}${image.image_url}`,
    };
  }
  return image;
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const message = typeof body?.message === "string" ? body.message.trim() : "";
  const documentId = typeof body?.documentId === "string" ? body.documentId : (body?.document_id ?? "");
  const includeImages = Boolean(body?.include_images ?? true);
  const topK = Number(body?.top_k ?? 8);
  
  const cookieStore = cookies();
  const domain = cookieStore.get("domain")?.value || "real-estate";

  if (!message) {
    return NextResponse.json({ error: "Message is required." }, { status: 400 });
  }

  try {
    const backendBody: Record<string, unknown> = {
      message,
      include_images: includeImages,
      top_k: topK,
      domain,
    };
    if (documentId) {
      backendBody.document_id = documentId;
    }

    const response = await fetch(`${backendBaseUrl}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(backendBody),
      cache: "no-store",
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: "The chatbot backend returned an error." },
        { status: response.status },
      );
    }

    const data = (await response.json()) as BackendChatResponse;

    // Resolve image URLs
    const resolvedImages = (data.images ?? []).map(resolveImage);

    return NextResponse.json({
      question: data.question ?? message,
      answer: data.answer ?? "No answer returned.",
      images: resolvedImages,
      citations: data.citations ?? [],
      confidence: data.confidence ?? "medium",
      intent: data.intent ?? {},
      retrieved_context: data.retrieved_context ?? [],
    });
  } catch {
    return NextResponse.json(
      {
        error:
          "Could not reach the chatbot backend. Start FastAPI on http://127.0.0.1:8000 and try again.",
      },
      { status: 503 },
    );
  }
}
