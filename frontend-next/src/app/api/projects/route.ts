import { NextResponse } from "next/server";
import { cookies } from "next/headers";

const backendBaseUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const cookieStore = cookies();
    const domain = cookieStore.get("domain")?.value || "real-estate";
    const response = await fetch(`${backendBaseUrl}/projects?domain=${domain}`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return NextResponse.json({ projects: [] }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ projects: [] });
  }
}
