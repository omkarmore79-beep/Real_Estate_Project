import { NextResponse } from "next/server";

const backendBaseUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function GET(
  _request: Request,
  { params }: { params: { document_id: string } },
) {
  const { document_id } = params;

  try {
    const response = await fetch(
      `${backendBaseUrl}/documents/${document_id}/status`,
      { cache: "no-store" },
    );

    if (response.status === 404) {
      return NextResponse.json(
        { error: "Document not found." },
        { status: 404 },
      );
    }

    if (!response.ok) {
      return NextResponse.json(
        { error: "Backend returned an error." },
        { status: response.status },
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "Could not reach the backend. Is FastAPI running?" },
      { status: 503 },
    );
  }
}
