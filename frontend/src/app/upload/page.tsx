"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { toast } from "react-toastify";
import api, { PredictionResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const schema = z.object({
  report_text: z.string().optional(),
  modality: z.enum(["T1", "T2", "STIR"]).default("T2"),
  sex: z.enum(["M", "F", "Other"]).optional(),
  age: z.coerce.number().min(0).max(150).optional(),
  height_cm: z.coerce.number().min(50).max(250).optional(),
  weight_kg: z.coerce.number().min(1).max(500).optional(),
});
type FormData = z.infer<typeof schema>;

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [step, setStep] = useState<"upload" | "details" | "processing" | "results">("upload");

  const { register, handleSubmit, formState: { errors } } = useForm<FormData>({
    resolver: zodResolver(schema),
    defaultValues: { modality: "T2" },
  });

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted[0]) {
      setFile(accepted[0]);
      setStep("details");
      toast.info(`File selected: ${accepted[0].name}`);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/octet-stream": [".mha", ".mhd", ".nii", ".gz", ".dcm"],
      "image/png": [".png"],
      "image/jpeg": [".jpg", ".jpeg"],
    },
    maxFiles: 1,
    maxSize: 500 * 1024 * 1024,
  });

  const onSubmit = async (data: FormData) => {
    if (!file) return toast.error("Please select an MRI file");
    setIsUploading(true);
    setStep("processing");
    setUploadProgress(0);

    try {
      const res = await api.uploadAndPredict(
        file,
        {
          report_text: data.report_text,
          modality: data.modality,
          age: data.age,
          sex: data.sex,
          height_cm: data.height_cm,
          weight_kg: data.weight_kg,
        },
        (p) => setUploadProgress(p)
      );
      setResult(res);
      setStep("results");
      toast.success("Analysis complete!");
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Analysis failed");
      setStep("details");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Nav */}
      <nav className="bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-4">
        <a href="/dashboard" className="text-gray-500 hover:text-gray-700">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </a>
        <h1 className="font-bold text-gray-900">New MRI Study</h1>
      </nav>

      <main className="max-w-4xl mx-auto px-6 py-8">
        {/* Progress steps */}
        <div className="flex items-center gap-2 mb-8">
          {["upload", "details", "processing", "results"].map((s, i) => (
            <div key={s} className="flex items-center gap-2">
              <div className={cn(
                "w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition",
                step === s ? "bg-blue-600 text-white"
                : ["upload","details","processing","results"].indexOf(step) > i
                  ? "bg-green-500 text-white" : "bg-gray-200 text-gray-500"
              )}>
                {["upload","details","processing","results"].indexOf(step) > i ? "✓" : i + 1}
              </div>
              <span className="text-sm text-gray-600 capitalize hidden sm:block">{s}</span>
              {i < 3 && <div className="w-8 h-0.5 bg-gray-200" />}
            </div>
          ))}
        </div>

        {/* Step: Upload */}
        {(step === "upload" || step === "details") && (
          <div className="space-y-6">
            {/* Dropzone */}
            <div
              {...getRootProps()}
              className={cn(
                "border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition",
                isDragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-blue-400 bg-white",
                file ? "border-green-400 bg-green-50" : ""
              )}
            >
              <input {...getInputProps()} />
              {file ? (
                <div>
                  <div className="text-4xl mb-3">✅</div>
                  <p className="font-medium text-green-700">{file.name}</p>
                  <p className="text-sm text-gray-500 mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); setFile(null); setStep("upload"); }}
                    className="mt-3 text-xs text-red-500 hover:underline"
                  >
                    Remove file
                  </button>
                </div>
              ) : (
                <div>
                  <div className="text-4xl mb-3">🏥</div>
                  <p className="text-lg font-medium text-gray-700">
                    {isDragActive ? "Drop the MRI file here..." : "Drop your MRI file here"}
                  </p>
                  <p className="text-sm text-gray-500 mt-2">
                    Supports: .mha, .mhd, .nii, .nii.gz, .dcm, .png, .jpg
                  </p>
                  <p className="text-xs text-gray-400 mt-1">Max size: 500 MB</p>
                </div>
              )}
            </div>

            {/* Details form */}
            {step === "details" && (
              <form onSubmit={handleSubmit(onSubmit)} className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm space-y-5">
                <h2 className="font-semibold text-gray-900 text-lg">Study Details</h2>

                {/* Modality */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">MRI Modality</label>
                  <select {...register("modality")}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
                    <option value="T2">T2-weighted (Recommended)</option>
                    <option value="T1">T1-weighted</option>
                    <option value="STIR">STIR</option>
                  </select>
                </div>

                {/* Report text */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Radiology Report <span className="text-gray-400">(optional)</span>
                  </label>
                  <textarea
                    {...register("report_text")}
                    rows={3}
                    placeholder="e.g. Posterior disc bulge at L4-L5 causing mild spinal canal stenosis..."
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none resize-none"
                  />
                </div>

                {/* Demographics */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Patient Demographics <span className="text-gray-400">(optional)</span>
                  </label>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    <div>
                      <label className="text-xs text-gray-500">Sex</label>
                      <select {...register("sex")} className="w-full mt-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
                        <option value="">—</option>
                        <option value="F">Female</option>
                        <option value="M">Male</option>
                        <option value="Other">Other</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">Age (years)</label>
                      <input {...register("age")} type="number" min={0} max={150}
                        className="w-full mt-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                        placeholder="e.g. 55" />
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">Height (cm)</label>
                      <input {...register("height_cm")} type="number" min={50} max={250}
                        className="w-full mt-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                        placeholder="e.g. 170" />
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">Weight (kg)</label>
                      <input {...register("weight_kg")} type="number" min={1} max={500}
                        className="w-full mt-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                        placeholder="e.g. 75" />
                    </div>
                  </div>
                </div>

                <Button type="submit" loading={isUploading} className="w-full h-11">
                  🚀 Analyze MRI
                </Button>
              </form>
            )}
          </div>
        )}

        {/* Step: Processing */}
        {step === "processing" && (
          <div className="bg-white rounded-2xl border border-gray-100 p-12 shadow-sm text-center">
            <div className="text-5xl mb-6 animate-pulse">🧠</div>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">Analyzing MRI...</h2>
            <p className="text-gray-500 text-sm mb-6">
              Running segmentation, classification, and report generation
            </p>
            {uploadProgress < 100 && (
              <div>
                <div className="bg-gray-100 rounded-full h-2 mb-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                <p className="text-xs text-gray-400">Uploading: {uploadProgress}%</p>
              </div>
            )}
            {uploadProgress === 100 && (
              <div className="space-y-2 text-sm text-gray-600">
                <p>✅ Upload complete</p>
                <p className="animate-pulse">⏳ Running AI analysis...</p>
              </div>
            )}
          </div>
        )}

        {/* Step: Results */}
        {step === "results" && result && (
          <ResultsPanel result={result} onNewStudy={() => { setFile(null); setResult(null); setStep("upload"); }} />
        )}
      </main>
    </div>
  );
}

function ResultsPanel({ result, onNewStudy }: { result: PredictionResponse; onNewStudy: () => void }) {
  const { classification, severity, levels, segmentation, report, pfirrmann_grade, gradcam_b64, inference_time_ms } = result;

  const confidenceColor =
    classification.confidence >= 0.8 ? "text-green-600" :
    classification.confidence >= 0.6 ? "text-yellow-600" : "text-red-600";

  return (
    <div className="space-y-6">
      {/* Summary card */}
      <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-lg font-bold text-gray-900">Analysis Results</h2>
            {inference_time_ms && (
              <p className="text-xs text-gray-400">Inference: {inference_time_ms.toFixed(0)}ms</p>
            )}
          </div>
          <button onClick={onNewStudy} className="text-sm text-blue-600 hover:underline">
            + New Study
          </button>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gray-50 rounded-xl p-4">
            <div className="text-xs font-medium text-gray-500 mb-1">Diagnosis</div>
            <div className="font-bold text-gray-900 text-sm">{classification.disease_name.replace(/_/g," ")}</div>
            <div className={`text-xs font-semibold mt-0.5 ${confidenceColor}`}>
              {(classification.confidence * 100).toFixed(1)}% confidence
            </div>
          </div>
          <div className="bg-gray-50 rounded-xl p-4">
            <div className="text-xs font-medium text-gray-500 mb-1">Severity</div>
            <div className="font-bold text-gray-900 text-sm">{severity.name}</div>
          </div>
          <div className="bg-gray-50 rounded-xl p-4">
            <div className="text-xs font-medium text-gray-500 mb-1">Pfirrmann Grade</div>
            <div className="font-bold text-gray-900 text-sm">{pfirrmann_grade.toFixed(1)} / 5</div>
          </div>
          <div className="bg-gray-50 rounded-xl p-4">
            <div className="text-xs font-medium text-gray-500 mb-1">Affected Levels</div>
            <div className="font-bold text-gray-900 text-sm">
              {levels.affected.length > 0 ? levels.affected.join(", ") : "None"}
            </div>
          </div>
        </div>
      </div>

      {/* Visuals */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {segmentation.overlay_b64 && (
          <div className="bg-white rounded-2xl border border-gray-100 p-4 shadow-sm">
            <h3 className="font-semibold text-gray-900 mb-3 text-sm">Segmentation Overlay</h3>
            <img
              src={`data:image/png;base64,${segmentation.overlay_b64}`}
              alt="Segmentation"
              className="w-full rounded-lg"
            />
          </div>
        )}
        {gradcam_b64 && (
          <div className="bg-white rounded-2xl border border-gray-100 p-4 shadow-sm">
            <h3 className="font-semibold text-gray-900 mb-3 text-sm">Grad-CAM Heatmap</h3>
            <img
              src={`data:image/png;base64,${gradcam_b64}`}
              alt="Grad-CAM"
              className="w-full rounded-lg"
            />
          </div>
        )}
      </div>

      {/* Report */}
      <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
        <h3 className="font-semibold text-gray-900 mb-4">Clinical Report</h3>
        <div className="space-y-3">
          <div>
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Findings</span>
            <p className="text-sm text-gray-700 mt-1">{report.findings}</p>
          </div>
          <div>
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Impression</span>
            <p className="text-sm text-gray-700 mt-1">{report.impression}</p>
          </div>
          <div>
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Recommendation</span>
            <p className="text-sm text-gray-700 mt-1">{report.recommendation}</p>
          </div>
        </div>

        <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
          <p className="text-xs text-amber-700">
            ⚠️ AI-generated report. Must be reviewed by a qualified radiologist before clinical use.
          </p>
        </div>
      </div>

      {/* Disease probabilities */}
      <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
        <h3 className="font-semibold text-gray-900 mb-4">Disease Probability Distribution</h3>
        <div className="space-y-2">
          {Object.entries(classification.disease_probabilities).map(([name, prob]) => (
            <div key={name} className="flex items-center gap-3">
              <div className="w-40 text-sm text-gray-600 shrink-0">{name.replace(/_/g," ")}</div>
              <div className="flex-1 bg-gray-100 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all"
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
              <div className="w-12 text-xs text-gray-500 text-right">{(prob * 100).toFixed(1)}%</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
