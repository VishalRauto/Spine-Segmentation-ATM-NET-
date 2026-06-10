"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { toast } from "react-toastify";
import api, { Patient } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { cn, formatDate } from "@/lib/utils";

const schema = z.object({
  patient_code: z.string().min(1).max(50),
  first_name: z.string().optional(),
  last_name: z.string().optional(),
  sex: z.enum(["M", "F", "Other"]).optional(),
  age: z.coerce.number().min(0).max(150).optional(),
  height_cm: z.coerce.number().min(50).max(250).optional(),
  weight_kg: z.coerce.number().min(1).max(500).optional(),
  clinical_symptoms: z.string().optional(),
});
type FormData = z.infer<typeof schema>;

export default function PatientsPage() {
  const router = useRouter();
  const qc = useQueryClient();
  const [search, setSearch] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [selected, setSelected] = useState<Patient | null>(null);

  const { data: patients = [], isLoading } = useQuery<Patient[]>({
    queryKey: ["patients", search],
    queryFn: () => api.listPatients(0, 100, search),
  });

  const createMutation = useMutation({
    mutationFn: (data: FormData) => api.createPatient(data),
    onSuccess: () => {
      toast.success("Patient created");
      qc.invalidateQueries({ queryKey: ["patients"] });
      setShowForm(false);
      reset();
    },
    onError: (err: any) => toast.error(err.response?.data?.detail || "Failed"),
  });

  const { register, handleSubmit, reset, formState: { errors } } = useForm<FormData>({
    resolver: zodResolver(schema),
  });

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <a href="/dashboard" className="text-gray-500 hover:text-gray-700">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7"/>
            </svg>
          </a>
          <h1 className="font-bold text-gray-900">Patient Records</h1>
        </div>
        <Button onClick={() => setShowForm(true)} size="sm">
          + Add Patient
        </Button>
      </nav>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {/* Search */}
        <div className="flex gap-3 mb-6">
          <div className="flex-1 relative">
            <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
            </svg>
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search by name or ID..."
              className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Patient table */}
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-100">
                {["Patient ID","Name","Sex","Age","BMI","Symptoms","Actions"].map(h => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr><td colSpan={7} className="text-center py-12 text-gray-400">Loading...</td></tr>
              ) : patients.length === 0 ? (
                <tr><td colSpan={7} className="text-center py-12 text-gray-400">
                  No patients found. <button onClick={() => setShowForm(true)} className="text-blue-500 hover:underline">Add one →</button>
                </td></tr>
              ) : patients.map(p => (
                <tr key={p.id} className="border-b border-gray-50 hover:bg-gray-50 transition">
                  <td className="px-4 py-3 text-sm font-mono font-medium text-blue-600">{p.patient_code}</td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {[p.first_name, p.last_name].filter(Boolean).join(" ") || "—"}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">{p.sex || "—"}</td>
                  <td className="px-4 py-3 text-sm text-gray-600">{p.age ? `${p.age}y` : "—"}</td>
                  <td className="px-4 py-3 text-sm text-gray-600">{p.bmi ? `${p.bmi.toFixed(1)}` : "—"}</td>
                  <td className="px-4 py-3 text-sm text-gray-500 max-w-[200px] truncate">
                    {p.clinical_symptoms || "—"}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      <button
                        onClick={() => router.push(`/upload?patient_id=${p.id}`)}
                        className="text-xs bg-blue-50 text-blue-700 px-3 py-1 rounded-lg hover:bg-blue-100 font-medium"
                      >
                        New Study
                      </button>
                      <button
                        onClick={() => setSelected(p)}
                        className="text-xs bg-gray-50 text-gray-600 px-3 py-1 rounded-lg hover:bg-gray-100"
                      >
                        View
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Stats */}
        <p className="text-xs text-gray-400 mt-3">{patients.length} patient(s) found</p>
      </main>

      {/* Add Patient Modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-lg">
            <div className="p-6 border-b border-gray-100 flex items-center justify-between">
              <h2 className="font-semibold text-gray-900">Add New Patient</h2>
              <button onClick={() => setShowForm(false)} className="text-gray-400 hover:text-gray-600">✕</button>
            </div>
            <form onSubmit={handleSubmit(d => createMutation.mutate(d))} className="p-6 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="col-span-2">
                  <label className="text-xs font-medium text-gray-500">Patient ID *</label>
                  <input {...register("patient_code")}
                    className="w-full mt-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                    placeholder="e.g. PT-001" />
                  {errors.patient_code && <p className="text-red-500 text-xs mt-1">{errors.patient_code.message}</p>}
                </div>
                <div>
                  <label className="text-xs font-medium text-gray-500">First Name</label>
                  <input {...register("first_name")}
                    className="w-full mt-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"/>
                </div>
                <div>
                  <label className="text-xs font-medium text-gray-500">Last Name</label>
                  <input {...register("last_name")}
                    className="w-full mt-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"/>
                </div>
                <div>
                  <label className="text-xs font-medium text-gray-500">Sex</label>
                  <select {...register("sex")} className="w-full mt-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
                    <option value="">—</option>
                    <option value="F">Female</option>
                    <option value="M">Male</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs font-medium text-gray-500">Age</label>
                  <input {...register("age")} type="number"
                    className="w-full mt-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"/>
                </div>
                <div>
                  <label className="text-xs font-medium text-gray-500">Height (cm)</label>
                  <input {...register("height_cm")} type="number"
                    className="w-full mt-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"/>
                </div>
                <div>
                  <label className="text-xs font-medium text-gray-500">Weight (kg)</label>
                  <input {...register("weight_kg")} type="number"
                    className="w-full mt-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"/>
                </div>
                <div className="col-span-2">
                  <label className="text-xs font-medium text-gray-500">Clinical Symptoms</label>
                  <textarea {...register("clinical_symptoms")} rows={2}
                    className="w-full mt-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none resize-none"
                    placeholder="e.g. Lower back pain, radiating to left leg"/>
                </div>
              </div>
              <div className="flex gap-3 pt-2">
                <Button type="button" variant="outline" className="flex-1" onClick={() => setShowForm(false)}>Cancel</Button>
                <Button type="submit" loading={createMutation.isPending} className="flex-1">Create Patient</Button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Patient detail modal */}
      {selected && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold text-gray-900">Patient: {selected.patient_code}</h2>
              <button onClick={() => setSelected(null)} className="text-gray-400 hover:text-gray-600">✕</button>
            </div>
            <dl className="space-y-2 text-sm">
              {[
                ["Full Name", [selected.first_name, selected.last_name].filter(Boolean).join(" ") || "—"],
                ["Sex", selected.sex || "—"],
                ["Age", selected.age ? `${selected.age} years` : "—"],
                ["BMI", selected.bmi ? `${selected.bmi.toFixed(1)} kg/m²` : "—"],
                ["Height", selected.height_cm ? `${selected.height_cm} cm` : "—"],
                ["Weight", selected.weight_kg ? `${selected.weight_kg} kg` : "—"],
                ["Symptoms", selected.clinical_symptoms || "None reported"],
                ["Created", formatDate(selected.created_at)],
              ].map(([label, value]) => (
                <div key={label as string} className="flex justify-between">
                  <dt className="text-gray-500">{label}</dt>
                  <dd className="font-medium text-gray-800 text-right max-w-[200px]">{value}</dd>
                </div>
              ))}
            </dl>
            <div className="flex gap-2 mt-6">
              <Button variant="outline" className="flex-1" onClick={() => setSelected(null)}>Close</Button>
              <Button className="flex-1" onClick={() => { router.push(`/upload?patient_id=${selected.id}`); setSelected(null); }}>
                New Study
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
