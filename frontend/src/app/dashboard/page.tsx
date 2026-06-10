"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from "recharts";
import api, { AnalyticsSummary } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import { formatDate, getSeverityColor } from "@/lib/utils";

const DISEASE_COLORS = ["#22c55e","#ef4444","#f97316","#a855f7","#eab308","#ec4899","#6b7280"];
const SEVERITY_COLORS = { Mild: "#22c55e", Moderate: "#f59e0b", Severe: "#ef4444" };

function StatCard({ title, value, subtitle, icon, color }: any) {
  return (
    <div className={`bg-white rounded-2xl border border-gray-100 p-6 shadow-sm`}>
      <div className="flex items-center justify-between mb-3">
        <div className={`p-2.5 rounded-xl ${color}`}>
          <span className="text-2xl">{icon}</span>
        </div>
      </div>
      <div className="text-3xl font-bold text-gray-900">{value}</div>
      <div className="text-sm font-medium text-gray-700 mt-0.5">{title}</div>
      {subtitle && <div className="text-xs text-gray-500 mt-1">{subtitle}</div>}
    </div>
  );
}

export default function DashboardPage() {
  const router = useRouter();
  const { user, isAuthenticated } = useAuthStore();

  useEffect(() => {
    if (!isAuthenticated) router.push("/auth/login");
  }, [isAuthenticated, router]);

  const { data: analytics, isLoading } = useQuery<AnalyticsSummary>({
    queryKey: ["analytics"],
    queryFn: () => api.getAnalyticsSummary(),
    refetchInterval: 30_000,
  });

  const diseaseChartData = analytics
    ? Object.entries(analytics.disease_distribution).map(([name, value]) => ({
        name: name.replace(/_/g, " "),
        value,
      }))
    : [];

  const severityChartData = analytics
    ? Object.entries(analytics.severity_distribution).map(([name, value]) => ({
        name, value,
        fill: SEVERITY_COLORS[name as keyof typeof SEVERITY_COLORS] || "#6b7280",
      }))
    : [];

  if (!isAuthenticated) return null;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top nav */}
      <nav className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-xs">AI</span>
          </div>
          <span className="font-bold text-gray-900">ATM-Net++</span>
        </div>
        <div className="flex items-center gap-4">
          <nav className="flex gap-6 text-sm font-medium text-gray-600">
            <a href="/dashboard" className="text-blue-600">Dashboard</a>
            <a href="/upload" className="hover:text-blue-600">New Study</a>
            <a href="/patients" className="hover:text-blue-600">Patients</a>
            <a href="/analytics" className="hover:text-blue-600">Analytics</a>
          </nav>
          <div className="flex items-center gap-2 pl-4 border-l border-gray-200">
            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
              <span className="text-blue-700 font-semibold text-xs">
                {user?.username?.[0]?.toUpperCase()}
              </span>
            </div>
            <span className="text-sm text-gray-700">{user?.username}</span>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Clinical Dashboard</h1>
            <p className="text-gray-500 text-sm mt-1">Lumbar Spine MRI Analysis Overview</p>
          </div>
          <button
            onClick={() => router.push("/upload")}
            className="flex items-center gap-2 bg-blue-600 text-white px-5 py-2.5 rounded-xl hover:bg-blue-700 font-medium text-sm transition"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Study
          </button>
        </div>

        {/* Stat cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard
            title="Total Studies"
            value={analytics?.total_studies ?? "—"}
            icon="🏥"
            color="bg-blue-50"
          />
          <StatCard
            title="Total Patients"
            value={analytics?.total_patients ?? "—"}
            icon="👥"
            color="bg-green-50"
          />
          <StatCard
            title="AI Predictions"
            value={analytics?.total_predictions ?? "—"}
            icon="🤖"
            color="bg-purple-50"
          />
          <StatCard
            title="Avg. Dice Score"
            value={analytics?.average_dice ? `${(analytics.average_dice * 100).toFixed(1)}%` : "—"}
            subtitle={analytics?.average_dice && analytics.average_dice > 0.90 ? "✅ Target achieved" : "Target: >90%"}
            icon="🎯"
            color="bg-amber-50"
          />
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Disease distribution bar chart */}
          <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
            <h2 className="font-semibold text-gray-900 mb-4">Disease Distribution</h2>
            {diseaseChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={diseaseChartData} margin={{ top: 0, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {diseaseChartData.map((_, i) => (
                      <Cell key={i} fill={DISEASE_COLORS[i % DISEASE_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[220px] flex items-center justify-center text-gray-400 text-sm">
                No prediction data yet
              </div>
            )}
          </div>

          {/* Severity pie chart */}
          <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
            <h2 className="font-semibold text-gray-900 mb-4">Severity Distribution</h2>
            {severityChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={severityChartData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {severityChartData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[220px] flex items-center justify-center text-gray-400 text-sm">
                No prediction data yet
              </div>
            )}
          </div>
        </div>

        {/* Quick actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            {
              title: "Upload MRI", desc: "Analyze new lumbar spine MRI scan",
              href: "/upload", icon: "📤", color: "border-blue-200 hover:border-blue-400"
            },
            {
              title: "Patient Records", desc: "View and manage patient database",
              href: "/patients", icon: "📋", color: "border-green-200 hover:border-green-400"
            },
            {
              title: "Analytics", desc: "Model performance and population stats",
              href: "/analytics", icon: "📊", color: "border-purple-200 hover:border-purple-400"
            },
          ].map((item) => (
            <a
              key={item.href}
              href={item.href}
              className={`bg-white border-2 rounded-2xl p-5 transition cursor-pointer ${item.color} block`}
            >
              <div className="text-3xl mb-3">{item.icon}</div>
              <div className="font-semibold text-gray-900">{item.title}</div>
              <div className="text-sm text-gray-500 mt-0.5">{item.desc}</div>
            </a>
          ))}
        </div>
      </main>
    </div>
  );
}
