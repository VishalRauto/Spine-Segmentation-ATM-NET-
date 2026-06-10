"use client";

import { useQuery } from "@tanstack/react-query";
import api from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function AdminPage() {
  const router = useRouter();
  const { user, isAuthenticated } = useAuthStore();

  useEffect(() => {
    if (!isAuthenticated) router.push("/auth/login");
    else if (user?.role !== "admin") router.push("/dashboard");
  }, [isAuthenticated, user, router]);

  const { data: performance } = useQuery({
    queryKey: ["model-performance"],
    queryFn: () => api.getModelPerformance(),
  });

  if (!isAuthenticated || user?.role !== "admin") return null;

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-3">
        <a href="/dashboard" className="text-gray-500 hover:text-gray-700">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7"/>
          </svg>
        </a>
        <h1 className="font-bold text-gray-900">Admin Panel</h1>
        <span className="ml-2 text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded-full font-medium">Admin Only</span>
      </nav>

      <main className="max-w-5xl mx-auto px-6 py-8 space-y-6">
        <h2 className="text-xl font-bold text-gray-900">System Administration</h2>

        {/* Model info */}
        <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
          <h3 className="font-semibold text-gray-800 mb-4">Model Performance</h3>
          {performance ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                ["Total Predictions", performance.total_predictions],
                ["Avg Dice", performance.average_dice ? `${(performance.average_dice * 100).toFixed(1)}%` : "N/A"],
                ["Avg Confidence", performance.average_confidence ? `${(performance.average_confidence * 100).toFixed(1)}%` : "N/A"],
                ["Avg Inference", performance.average_inference_ms ? `${performance.average_inference_ms.toFixed(0)}ms` : "N/A"],
              ].map(([label, value]) => (
                <div key={label as string} className="bg-gray-50 rounded-xl p-4">
                  <div className="text-xs text-gray-500 mb-1">{label}</div>
                  <div className="text-xl font-bold text-gray-900">{value}</div>
                </div>
              ))}
            </div>
          ) : <p className="text-gray-500 text-sm">Loading...</p>}
        </div>

        {/* System actions */}
        <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
          <h3 className="font-semibold text-gray-800 mb-4">System Actions</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {[
              { label: "Health Check", desc: "Verify all services are running", icon: "💚", href: "http://localhost:8000/health" },
              { label: "API Docs",     desc: "View Swagger documentation",      icon: "📖", href: "http://localhost:8000/docs" },
              { label: "API Redoc",    desc: "View ReDoc documentation",        icon: "📋", href: "http://localhost:8000/redoc" },
            ].map(item => (
              <a
                key={item.label}
                href={item.href}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-start gap-3 p-4 border border-gray-200 rounded-xl hover:bg-gray-50 transition"
              >
                <span className="text-2xl">{item.icon}</span>
                <div>
                  <div className="font-medium text-sm text-gray-900">{item.label}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{item.desc}</div>
                </div>
              </a>
            ))}
          </div>
        </div>

        {/* Environment info */}
        <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
          <h3 className="font-semibold text-gray-800 mb-4">Environment</h3>
          <dl className="space-y-2 text-sm">
            {[
              ["API URL", process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1"],
              ["User", `${user?.username} (${user?.role})`],
              ["App Version", "ATM-Net++ v1.0.0"],
            ].map(([k, v]) => (
              <div key={k as string} className="flex justify-between">
                <dt className="text-gray-500">{k}</dt>
                <dd className="font-mono text-xs text-gray-700">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </main>
    </div>
  );
}
