import { NextResponse } from "next/server";

type BackendChatResponse = {
  question?: string;
  answer?: string;
  images?: string[];
};

const backendBaseUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

function toBackendImageUrl(imagePath: string) {
  if (imagePath.startsWith("http://") || imagePath.startsWith("https://")) {
    return imagePath;
  }

  return `${backendBaseUrl}/${imagePath.replace(/^\/+/, "")}`;
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const message = typeof body?.message === "string" ? body.message.trim() : "";
  const documentId = typeof body?.documentId === "string" ? body.documentId : "";

  if (!message) {
    return NextResponse.json({ error: "Message is required." }, { status: 400 });
  }

  try {
    const response = await fetch(`${backendBaseUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(documentId ? { message, document_id: documentId } : message),
      cache: "no-store",
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: "The chatbot backend returned an error." },
        { status: response.status },
      );
    }

    const data = (await response.json()) as BackendChatResponse;

    return NextResponse.json({
      question: data.question ?? message,
      answer: data.answer ?? "No answer returned.",
      images: (data.images ?? []).map(toBackendImageUrl),
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
