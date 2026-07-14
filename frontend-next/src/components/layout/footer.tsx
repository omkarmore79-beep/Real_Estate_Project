export function Footer() {
  return (
    <footer className="mt-auto border-t border-slate-200 bg-white py-6">
      <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-4 text-sm text-muted-foreground md:flex-row md:px-6">
        <p>&copy; {new Date().getFullYear()} RAG Knowledge Platform. All rights reserved.</p>
        <div className="flex gap-6">
          <a href="#" className="hover:text-foreground transition">Privacy Policy</a>
          <a href="#" className="hover:text-foreground transition">Terms of Service</a>
          <a href="#" className="hover:text-foreground transition">Support Helpdesk</a>
        </div>
      </div>
    </footer>
  );
}
