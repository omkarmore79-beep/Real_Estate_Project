"use client";

import Link from "next/link";
import { ArrowLeft, BotMessageSquare, FileText, Home, ImageIcon, Loader2, Send, Sparkles } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import { Badge } from "@/components/ui";
import { getBuilderName, getProjectTitle, type UploadedProject } from "@/lib/backend-data";
import { cn } from "@/lib/utils";

type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
  images?: string[];
};

const quickQuestions = [
  "What is the project name?",
  "Where is the project located?",
  "What amenities are available?",
  "Show me the floor plan",
  "What is the RERA number?",
  "Show me the location plan",
  "What is the possession date?",
  "What are the contact details?",
];

export default function UserChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 1,
      role: "assistant",
      content:
        "Hi, I can answer questions from the uploaded real estate brochures. Ask about amenities, location, floor plans, RERA, possession, or contact details.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [activeDocumentId, setActiveDocumentId] = useState("");
  const [documents, setDocuments] = useState<UploadedProject[]>([]);

  useEffect(() => {
    const documentIdFromUrl = new URLSearchParams(window.location.search).get("documentId");
    const nextDocumentId = documentIdFromUrl || localStorage.getItem("activeDocumentId") || "";
    setActiveDocumentId(nextDocumentId);
    if (nextDocumentId) {
      localStorage.setItem("activeDocumentId", nextDocumentId);
    }

    fetch("/api/projects")
      .then((response) => response.json())
      .then((data) => {
        setDocuments(Array.isArray(data.projects) ? data.projects : []);
      })
      .catch(() => setDocuments([]));
  }, []);

  const activeDocument = documents.find((document) => document.document_id === activeDocumentId);

  function handleDocumentChange(documentId: string) {
    setActiveDocumentId(documentId);
    if (documentId) {
      localStorage.setItem("activeDocumentId", documentId);
      window.history.replaceState(null, "", `/chat?documentId=${documentId}`);
    } else {
      localStorage.removeItem("activeDocumentId");
      window.history.replaceState(null, "", "/chat");
    }
  }

  async function sendMessage(messageText: string) {
    const cleanMessage = messageText.trim();
    if (!cleanMessage || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now(),
      role: "user",
      content: cleanMessage,
    };

    setMessages((current) => [...current, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: cleanMessage, documentId: activeDocumentId }),
      });
      const data = await response.json();

      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: "assistant",
          content: response.ok ? data.answer : data.error,
          images: response.ok ? data.images : [],
        },
      ]);
    } catch {
      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: "assistant",
          content: "Something went wrong while sending your question. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    sendMessage(input);
  }

  return (
    <main className="min-h-screen bg-[linear-gradient(180deg,#f8fafc_0%,#eef7f5_48%,#f8fafc_100%)]">
      <header className="border-b bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-4 md:px-6">
          <Link href="/" className="inline-flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-primary">
            <ArrowLeft className="h-4 w-4" />
            Admin panel
          </Link>
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <BotMessageSquare className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold">Estate Assistant</p>
              <p className="text-xs text-muted-foreground">Buyer chat</p>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-6 px-4 py-6 md:px-6 lg:grid-cols-[360px_1fr]">
        <aside className="space-y-4">
          <section className="panel overflow-hidden">
            <div className="bg-primary p-5 text-primary-foreground">
              <div className="flex items-center gap-3">
                <FileText className="h-6 w-6" />
                <div>
                  <h1 className="text-xl font-semibold">Find your next home</h1>
                  <p className="mt-1 text-sm text-primary-foreground/80">Ask the brochure in plain language.</p>
                </div>
              </div>
            </div>
            <div className="space-y-4 p-5">
              <div>
                <p className="text-sm text-muted-foreground">Active document</p>
                <h2 className="mt-1 font-semibold">
                  {activeDocument ? getProjectTitle(activeDocument) : "No uploaded document selected"}
                </h2>
                {activeDocument ? (
                  <p className="mt-1 text-xs text-muted-foreground">
                    {getBuilderName(activeDocument)} - {activeDocument.source_file || activeDocument.document_id}
                  </p>
                ) : null}
                <p className="mt-2 text-sm text-muted-foreground">
                  {activeDocumentId
                    ? "Questions are scoped to this selected PDF only."
                    : "Upload or select a PDF first to scope answers to a real file."}
                </p>
              </div>
              <label className="space-y-2">
                <span className="label">Ask from PDF</span>
                <select
                  className="field"
                  value={activeDocumentId}
                  onChange={(event) => handleDocumentChange(event.target.value)}
                >
                  <option value="">Select a PDF/project</option>
                  {documents.map((document, index) => (
                    <option
                      key={document.document_id ?? `${document.source_file ?? "document"}-${index}`}
                      value={document.document_id ?? ""}
                    >
                      {getBuilderName(document)} / {getProjectTitle(document)}
                    </option>
                  ))}
                </select>
              </label>
              <div className="flex flex-wrap gap-2">
                <Badge>{activeDocumentId ? "Real upload" : "Awaiting upload"}</Badge>
                <Badge>Scoped answers</Badge>
              </div>
              <div className="grid gap-2 text-sm">
                <div className="flex items-center gap-2 rounded-md bg-secondary px-3 py-2">
                  <Home className="h-4 w-4 text-primary" />
                  Project details, towers, and location
                </div>
                <div className="flex items-center gap-2 rounded-md bg-secondary px-3 py-2">
                  <ImageIcon className="h-4 w-4 text-primary" />
                  Floor plans and brochure images
                </div>
                <div className="flex items-center gap-2 rounded-md bg-secondary px-3 py-2">
                  <Sparkles className="h-4 w-4 text-primary" />
                  AI answers from uploaded documents
                </div>
              </div>
            </div>
          </section>

          <section className="panel p-5">
            <h2 className="font-semibold">Quick questions</h2>
            <div className="mt-4 grid gap-2">
              {quickQuestions.map((question) => (
                <button
                  key={question}
                  type="button"
                  onClick={() => sendMessage(question)}
                  className="rounded-md border bg-white px-3 py-2 text-left text-sm transition hover:border-primary/40 hover:bg-primary/5"
                  disabled={isLoading}
                >
                  {question}
                </button>
              ))}
            </div>
          </section>
        </aside>

        <section className="panel flex min-h-[calc(100vh-120px)] flex-col overflow-hidden">
          <div className="border-b bg-white px-5 py-4">
            <h2 className="font-semibold">Chat with AI</h2>
            <p className="text-sm text-muted-foreground">
              Answers come from the Python chatbot backend when it is running.
            </p>
          </div>

          <div className="flex-1 space-y-4 overflow-y-auto bg-secondary/40 p-4 md:p-5">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn("flex", message.role === "user" ? "justify-end" : "justify-start")}
              >
                <div
                  className={cn(
                    "max-w-[820px] rounded-lg px-4 py-3 text-sm shadow-sm",
                    message.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "border bg-white text-foreground",
                  )}
                >
                  <p className="whitespace-pre-wrap leading-6">{message.content}</p>
                  {message.images && message.images.length > 0 ? (
                    <div className="mt-3 grid gap-3 sm:grid-cols-2">
                      {message.images.map((image) => (
                        <img
                          key={image}
                          src={image}
                          alt="Related project document"
                          className="max-h-72 w-full rounded-md border object-contain"
                        />
                      ))}
                    </div>
                  ) : null}
                </div>
              </div>
            ))}
            {isLoading ? (
              <div className="flex justify-start">
                <div className="inline-flex items-center gap-2 rounded-lg border bg-white px-4 py-3 text-sm text-muted-foreground shadow-sm">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Thinking...
                </div>
              </div>
            ) : null}
          </div>

          <form onSubmit={handleSubmit} className="border-t bg-white p-4">
            <div className="flex gap-3">
              <input
                className="field h-11"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Ask about price, floor plan, amenities, RERA, possession..."
              />
              <button
                type="submit"
                className="inline-flex h-11 items-center gap-2 rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={isLoading || input.trim().length === 0}
              >
                <Send className="h-4 w-4" />
                Send
              </button>
            </div>
          </form>
        </section>
      </div>
    </main>
  );
}
