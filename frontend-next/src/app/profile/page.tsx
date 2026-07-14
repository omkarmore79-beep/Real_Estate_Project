"use client";

import { UserCircle, Mail, Briefcase, Shield, Save } from "lucide-react";
import { PageHeader } from "@/components/ui";

export default function ProfilePage() {
  return (
    <>
      <PageHeader 
        title="User Profile" 
        description="Manage your account settings and preferences." 
      />
      <div className="mx-auto max-w-3xl mt-6">
        <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
          
          <div className="bg-slate-50 p-6 sm:p-8 flex items-center gap-6 border-b border-slate-200">
            <div className="flex h-20 w-20 items-center justify-center rounded-full bg-primary/10 text-primary">
              <UserCircle className="h-12 w-12" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900">Admin User</h1>
              <p className="text-slate-500">System Administrator</p>
            </div>
          </div>

          <form className="p-6 sm:p-8 space-y-6" onSubmit={(e) => e.preventDefault()}>
            <div className="grid gap-6 sm:grid-cols-2">
              <label className="space-y-2">
                <span className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                  <UserCircle className="h-4 w-4 text-slate-400" /> Full Name
                </span>
                <input 
                  type="text" 
                  defaultValue="Admin User" 
                  className="h-10 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </label>

              <label className="space-y-2">
                <span className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                  <Mail className="h-4 w-4 text-slate-400" /> Email Address
                </span>
                <input 
                  type="email" 
                  defaultValue="admin@hyundai.com" 
                  className="h-10 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </label>

              <label className="space-y-2">
                <span className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                  <Briefcase className="h-4 w-4 text-slate-400" /> Department
                </span>
                <input 
                  type="text" 
                  defaultValue="Engineering" 
                  className="h-10 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </label>

              <label className="space-y-2">
                <span className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                  <Shield className="h-4 w-4 text-slate-400" /> Role
                </span>
                <input 
                  type="text" 
                  defaultValue="Administrator" 
                  disabled
                  className="h-10 w-full rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-500 cursor-not-allowed"
                />
              </label>
            </div>

            <div className="pt-4 flex justify-end">
              <button 
                type="button" 
                className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-primary/90 transition"
              >
                <Save className="h-4 w-4" /> Save Changes
              </button>
            </div>
          </form>
        </div>
      </div>
    </>
  );
}
