import { NextResponse } from "next/server";

const backendBaseUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function GET(
  _request: Request,
  { params }: { params: { document_id: string } },
) {
  try {
    const { document_id } = params;
    const response = await fetch(`${backendBaseUrl}/documents/${document_id}/pdf`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return NextResponse.json({ error: "PDF not found." }, { status: response.status });
    }

    return new Response(await response.arrayBuffer(), {
      headers: {
        "Content-Type": response.headers.get("Content-Type") ?? "application/pdf",
        "Content-Disposition": "inline",
      },
    });
  } catch {
    return NextResponse.json(
      {
        error:
          "Could not reach the backend. Start FastAPI on http://127.0.0.1:8000 and try again.",
      },
      { status: 503 },
    );
  }
}

export async function DELETE(
  _request: Request,
  { params }: { params: { document_id: string } },
) {
  try {
    const { document_id } = params;
    const response = await fetch(`${backendBaseUrl}/documents/${document_id}`, {
      method: "DELETE",
      cache: "no-store",
    });

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      return NextResponse.json(
        { error: data?.detail ?? "Could not delete this document." },
        { status: response.status },
      );
    }

    return NextResponse.json(data ?? { message: "Document deleted" });
  } catch {
    return NextResponse.json(
      {
        error:
          "Could not reach the backend. Start FastAPI on http://127.0.0.1:8000 and try again.",
      },
      { status: 503 },
    );
  }
}
