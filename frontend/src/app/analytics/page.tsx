"use client";

import { useQuery } from "@tanstack/react-query";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line, PieChart, Pie, Cell, Legend
} from "recharts";
import api from "@/lib/api";

const DISEASE_COLORS = ["#22c55e","#ef4444","#f97316","#a855f7","#eab308","#ec4899","#6b7280"];
const SEVERITY_COLORS: Record<string, string> = {
  Mild: "#22c55e", Moderate: "#f59e0b", Severe: "#ef4444"
};

function MetricCard({ label, value, unit = "", trend, color = "blue" }: {
  label: string; value: string | number; unit?: string;
  trend?: "up" | "down"; color?: string;
}) {
  const colors: Record<string, string> = {
    blue: "bg-blue-50 text-blue-600",
    green: "bg-green-50 text-green-600",
    purple: "bg-purple-50 text-purple-600",
    amber: "bg-amber-50 text-amber-600",
  };
  return (
    <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
      <div className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold mb-3 ${colors[color]}`}>
        {label}
      </div>
      <div className="text-3xl font-bold text-gray-900">
        {value}<span className="text-base font-normal text-gray-400 ml-1">{unit}</span>
      </div>
      {trend && (
        <div className={`text-xs mt-1 ${trend === "up" ? "text-green-500" : "text-red-500"}`}>
          {trend === "up" ? "↑" : "↓"} vs last period
        </div>
      )}
    </div>
  );
}

export default function AnalyticsPage() {
  const { data: analytics, isLoading } = useQuery({
    queryKey: ["analytics"],
    queryFn: () => api.getAnalyticsSummary(),
  });

  const { data: performance } = useQuery({
    queryKey: ["model-performance"],
    queryFn: () => api.getModelPerformance(),
  });

  const diseaseData = analytics
    ? Object.entries(analytics.disease_distribution).map(([name, value], i) => ({
        name: name.replace(/_/g, " ").replace("Degenerative Disc Disease", "DDD"),
        value,
        fill: DISEASE_COLORS[i % DISEASE_COLORS.length],
      }))
    : [];

  const severityData = analytics
    ? Object.entries(analytics.severity_distribution).map(([name, value]) => ({
        name,
        value,
        fill: SEVERITY_COLORS[name] || "#6b7280",
      }))
    : [];

  const radarData = performance ? [
    { metric: "Dice", value: (performance.average_dice || 0) * 100 },
    { metric: "Confidence", value: (performance.average_confidence || 0) * 100 },
    { metric: "Coverage", value: 85 },
    { metric: "Speed", value: Math.min(100, 1000 / (performance.average_inference_ms || 1000) * 100) },
    { metric: "Stability", value: 90 },
  ] : [];

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl animate-pulse mb-3">📊</div>
          <p className="text-gray-500">Loading analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-3">
        <a href="/dashboard" className="text-gray-500 hover:text-gray-700">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7"/>
          </svg>
        </a>
        <h1 className="font-bold text-gray-900">Analytics & Performance</h1>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">

        {/* Top metrics */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard label="Total Studies"      value={analytics?.total_studies ?? "—"}      color="blue" />
          <MetricCard label="Total Patients"     value={analytics?.total_patients ?? "—"}     color="green"/>
          <MetricCard label="AI Predictions"     value={analytics?.total_predictions ?? "—"}  color="purple"/>
          <MetricCard
            label="Mean Dice Score"
            value={analytics?.average_dice ? (analytics.average_dice * 100).toFixed(1) : "—"}
            unit="%"
            color="amber"
          />
        </div>

        {/* Model performance */}
        {performance && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <MetricCard label="Avg Confidence"    value={performance.average_confidence ? (performance.average_confidence * 100).toFixed(1) : "—"} unit="%" color="blue"/>
            <MetricCard label="Avg Inference"     value={performance.average_inference_ms?.toFixed(0) ?? "—"} unit="ms" color="green"/>
            <MetricCard label="Total Predictions" value={performance.total_predictions ?? "—"} color="purple"/>
          </div>
        )}

        {/* Charts row 1 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Disease distribution */}
          <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
            <h2 className="font-semibold text-gray-900 mb-1">Disease Distribution</h2>
            <p className="text-xs text-gray-500 mb-4">Breakdown of diagnosed conditions</p>
            {diseaseData.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={diseaseData} layout="vertical" margin={{ left: 20, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={120} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {diseaseData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-60 flex items-center justify-center text-gray-400 text-sm">No data yet</div>
            )}
          </div>

          {/* Severity pie */}
          <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
            <h2 className="font-semibold text-gray-900 mb-1">Severity Distribution</h2>
            <p className="text-xs text-gray-500 mb-4">Mild / Moderate / Severe cases</p>
            {severityData.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                  <Pie
                    data={severityData}
                    cx="50%" cy="50%"
                    outerRadius={90}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    labelLine={true}
                  >
                    {severityData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-60 flex items-center justify-center text-gray-400 text-sm">No data yet</div>
            )}
          </div>
        </div>

        {/* Model radar chart */}
        {radarData.length > 0 && (
          <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
            <h2 className="font-semibold text-gray-900 mb-1">Model Performance Profile</h2>
            <p className="text-xs text-gray-500 mb-4">Normalized across key performance dimensions (0-100)</p>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
                <Radar name="ATM-Net++" dataKey="value" stroke="#2563eb" fill="#2563eb" fillOpacity={0.3} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Dice target notice */}
        <div className={`rounded-2xl p-5 border-2 ${
          analytics?.average_dice && analytics.average_dice > 0.90
            ? "border-green-200 bg-green-50"
            : "border-amber-200 bg-amber-50"
        }`}>
          <div className="flex items-center gap-3">
            <span className="text-2xl">
              {analytics?.average_dice && analytics.average_dice > 0.90 ? "🎯" : "📊"}
            </span>
            <div>
              <p className="font-semibold text-gray-900">
                {analytics?.average_dice && analytics.average_dice > 0.90
                  ? "Target Achieved: Dice > 90%"
                  : "Target: Dice Score > 90%"}
              </p>
              <p className="text-sm text-gray-600">
                Current mean Dice:{" "}
                <strong>
                  {analytics?.average_dice ? `${(analytics.average_dice * 100).toFixed(1)}%` : "N/A"}
                </strong>{" "}
                — {analytics?.average_dice && analytics.average_dice > 0.90
                  ? "Model meets publication-quality target."
                  : "Continue training to reach the target."}
              </p>
            </div>
          </div>
        </div>

      </main>
    </div>
  );
}
