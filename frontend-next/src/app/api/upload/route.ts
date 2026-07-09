import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  // Parse the incoming multipart form from the browser
  let formData: FormData;
  try {
    formData = await request.formData();
  } catch (err) {
    console.error("[UPLOAD] Failed to parse form data:", err);
    return NextResponse.json(
      { error: "Could not parse form data. Make sure you are sending multipart/form-data." },
      { status: 400 },
    );
  }

  // Validate file is present and is a File (not a plain string)
  const file = formData.get("file");
  if (!(file instanceof File)) {
    console.error("[UPLOAD] 'file' field missing or not a File object");
    return NextResponse.json(
      { error: "No PDF file was included. Append it as formData.append('file', pdfFile)." },
      { status: 400 },
    );
  }

  console.log(`[UPLOAD] Forwarding to backend: ${file.name} (${file.size} bytes)`);
  console.log(`[UPLOAD] Backend URL: ${BACKEND_URL}/upload`);

  // Forward the FormData to FastAPI
  // Do NOT set Content-Type manually — fetch sets it automatically with the boundary
  let backendResponse: Response;
  try {
    backendResponse = await fetch(`${BACKEND_URL}/upload`, {
      method: "POST",
      body: formData,
      // No Content-Type header — fetch adds multipart boundary automatically
    });
  } catch (err) {
    console.error("[UPLOAD] Could not reach backend:", err);
    return NextResponse.json(
      {
        error: `Could not reach the backend at ${BACKEND_URL}. Make sure FastAPI is running.`,
      },
      { status: 503 },
    );
  }

  // Parse backend response
  let data: unknown = null;
  const contentType = backendResponse.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    data = await backendResponse.json().catch((err) => {
      console.error("[UPLOAD] Failed to parse backend JSON:", err);
      return null;
    });
  } else {
    const text = await backendResponse.text().catch(() => "");
    console.error("[UPLOAD] Backend returned non-JSON:", text.slice(0, 500));
    data = { error: "Backend returned an unexpected response format." };
  }

  console.log(`[UPLOAD] Backend status: ${backendResponse.status}`, data);

  if (!backendResponse.ok) {
    const errData = data as Record<string, unknown> | null;
    return NextResponse.json(
      {
        error:
          (errData as Record<string, string> | null)?.detail ??
          (errData as Record<string, string> | null)?.error ??
          `Upload failed (HTTP ${backendResponse.status})`,
      },
      { status: backendResponse.status },
    );
  }

  return NextResponse.json(data ?? { message: "Uploaded successfully." });
}
