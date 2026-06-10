import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(date: string | Date): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  }).format(new Date(date));
}

export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}

export function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return "text-green-600";
  if (confidence >= 0.6) return "text-yellow-600";
  return "text-red-600";
}

export function getSeverityColor(severity: string): string {
  switch (severity.toLowerCase()) {
    case "mild":     return "text-green-600 bg-green-50 border-green-200";
    case "moderate": return "text-yellow-600 bg-yellow-50 border-yellow-200";
    case "severe":   return "text-red-600 bg-red-50 border-red-200";
    default:         return "text-gray-600 bg-gray-50 border-gray-200";
  }
}

export function getDiseaseColor(disease: string): string {
  const colors: Record<string, string> = {
    Normal:                       "text-green-700 bg-green-50",
    Disc_Herniation:              "text-red-700 bg-red-50",
    Disc_Bulge:                   "text-orange-700 bg-orange-50",
    Spinal_Stenosis:              "text-purple-700 bg-purple-50",
    Degenerative_Disc_Disease:    "text-yellow-700 bg-yellow-50",
    Spondylolisthesis:            "text-pink-700 bg-pink-50",
    Compression_Fracture:         "text-red-800 bg-red-100",
  };
  return colors[disease] || "text-gray-700 bg-gray-50";
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
