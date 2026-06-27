import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatFileSize(sizeInMb: number) {
  return `${sizeInMb.toFixed(1)} MB`;
}
