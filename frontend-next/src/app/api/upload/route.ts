import { NextResponse } from "next/server";

const backendBaseUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  const formData = await request.formData();
  const file = formData.get("file");

  if (!(file instanceof File)) {
    return NextResponse.json({ error: "File is required." }, { status: 400 });
  }

  let response: Response;

  try {
    response = await fetch(`${backendBaseUrl}/upload`, {
      method: "POST",
      body: formData,
      cache: "no-store",
    });
  } catch (error) {
    console.error("UPLOAD BACKEND FETCH ERROR:", error);
    return NextResponse.json(
      {
        error:
          "Could not reach the ingestion backend. Start FastAPI on http://127.0.0.1:8000 and try again.",
      },
      { status: 503 },
    );
  }

  const data = await response.json().catch((error) => {
    console.error("UPLOAD BACKEND JSON ERROR:", error);
    return null;
  });

  if (!response.ok) {
    return NextResponse.json(
      { error: data?.detail ?? "The backend could not process this file." },
      { status: response.status },
    );
  }

  if (!data) {
    return NextResponse.json({
      message: "Processed successfully",
      saved_to_mongodb: true,
    });
  }

  return NextResponse.json(data);
}
