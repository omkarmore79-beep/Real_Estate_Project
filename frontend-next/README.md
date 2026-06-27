# EstateDocs Admin

Modern Next.js frontend for a real estate document management chatbot/admin system.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Project structure

```text
frontend-next/
  src/app/                  App Router pages
  src/app/page.tsx          Dashboard
  src/app/builders/         Builder list and add/edit screens
  src/app/projects/         Project list, add/edit, and detail screens
  src/app/documents/        Documents table and upload screen
  src/app/search/           Search and filter screen
  src/app/settings/         Settings placeholder
  src/components/           Reusable shell, forms, tables, and UI helpers
  src/lib/                  Mock data, types, and utilities
```

## Notes

- The UI uses mock data only.
- Upload buttons and form submissions are frontend-only placeholders.
- Comments marked `TODO` show where backend/API integration should be added later.
