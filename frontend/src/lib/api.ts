/**
 * ATM-Net++ API client with full TypeScript types.
 * Centralizes all HTTP requests to the FastAPI backend.
 */

import axios, { AxiosInstance, AxiosRequestConfig } from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface User {
  id: string;
  email: string;
  username: string;
  full_name: string | null;
  role: string;
  is_active: boolean;
  created_at: string;
}

export interface Patient {
  id: string;
  patient_code: string;
  first_name: string | null;
  last_name: string | null;
  sex: string | null;
  age: number | null;
  height_cm: number | null;
  weight_kg: number | null;
  bmi: number | null;
  clinical_symptoms: string | null;
  created_at: string;
}

export interface Study {
  id: string;
  study_uid: string;
  patient_id: string;
  modality: string;
  status: "pending" | "processing" | "completed" | "failed" | "reviewed";
  image_filename: string | null;
  radiology_report: string | null;
  created_at: string;
  processed_at: string | null;
}

export interface SegmentationResult {
  overlay_b64: string;
  class_distribution: Record<string, number>;
  detected_structures: string[];
}

export interface ClassificationResult {
  disease_id: number;
  disease_name: string;
  confidence: number;
  disease_probabilities: Record<string, number>;
}

export interface SeverityResult {
  id: number;
  name: "Mild" | "Moderate" | "Severe";
}

export interface LevelResult {
  affected: string[];
  all_probs: Record<string, number>;
}

export interface ReportResult {
  report_text: string;
  findings: string;
  impression: string;
  recommendation: string;
  disease_name: string;
  severity: string;
  affected_levels: string[];
  confidence: number;
  pfirrmann_grade: number;
}

export interface PredictionResponse {
  id: string | null;
  study_id: string | null;
  segmentation: SegmentationResult;
  classification: ClassificationResult;
  severity: SeverityResult;
  levels: LevelResult;
  pfirrmann_grade: number;
  report: ReportResult;
  gradcam_b64: string | null;
  inference_time_ms: number | null;
  num_slices_processed: number | null;
  model_version: string;
}

export interface AnalyticsSummary {
  total_studies: number;
  total_patients: number;
  total_predictions: number;
  disease_distribution: Record<string, number>;
  severity_distribution: Record<string, number>;
  average_dice: number | null;
  average_inference_time_ms: number | null;
}

// ─────────────────────────────────────────────────────────────────────
// API Client
// ─────────────────────────────────────────────────────────────────────

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 120_000, // 2 min for large MRI files
      headers: { "Content-Type": "application/json" },
    });

    // Request interceptor: attach JWT
    this.client.interceptors.request.use((config) => {
      const token = this.getToken();
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Response interceptor: handle 401 → refresh token
    this.client.interceptors.response.use(
      (res) => res,
      async (error) => {
        if (error.response?.status === 401) {
          const refreshed = await this.refreshAccessToken();
          if (refreshed) {
            error.config.headers.Authorization = `Bearer ${this.getToken()}`;
            return this.client.request(error.config);
          } else {
            this.clearTokens();
            if (typeof window !== "undefined") {
              window.location.href = "/auth/login";
            }
          }
        }
        return Promise.reject(error);
      }
    );
  }

  private getToken(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("access_token");
  }

  private getRefreshToken(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("refresh_token");
  }

  private setTokens(access: string, refresh: string) {
    localStorage.setItem("access_token", access);
    localStorage.setItem("refresh_token", refresh);
  }

  private clearTokens() {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    localStorage.removeItem("user");
  }

  private async refreshAccessToken(): Promise<boolean> {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken) return false;
    try {
      const res = await axios.post(`${API_BASE_URL}/auth/refresh`, null, {
        params: { refresh_token: refreshToken },
      });
      this.setTokens(res.data.access_token, res.data.refresh_token);
      return true;
    } catch {
      return false;
    }
  }

  // ── Auth ──────────────────────────────────────────────────────────

  async login(username: string, password: string): Promise<{ token: TokenResponse; user: User }> {
    const res = await this.client.post<TokenResponse>("/auth/login", { username, password });
    this.setTokens(res.data.access_token, res.data.refresh_token);
    const user = await this.getMe();
    localStorage.setItem("user", JSON.stringify(user));
    return { token: res.data, user };
  }

  async register(data: { email: string; username: string; full_name?: string; password: string }): Promise<User> {
    const res = await this.client.post<User>("/auth/register", data);
    return res.data;
  }

  async getMe(): Promise<User> {
    const res = await this.client.get<User>("/auth/me");
    return res.data;
  }

  logout() {
    this.clearTokens();
  }

  // ── Patients ──────────────────────────────────────────────────────

  async createPatient(data: Partial<Patient>): Promise<Patient> {
    const res = await this.client.post<Patient>("/patients", data);
    return res.data;
  }

  async listPatients(skip = 0, limit = 50, search = ""): Promise<Patient[]> {
    const res = await this.client.get<Patient[]>("/patients", {
      params: { skip, limit, search },
    });
    return res.data;
  }

  async getPatient(id: string): Promise<Patient> {
    const res = await this.client.get<Patient>(`/patients/${id}`);
    return res.data;
  }

  async updatePatient(id: string, data: Partial<Patient>): Promise<Patient> {
    const res = await this.client.put<Patient>(`/patients/${id}`, data);
    return res.data;
  }

  // ── Prediction ────────────────────────────────────────────────────

  async uploadAndPredict(
    file: File,
    options: {
      report_text?: string;
      modality?: string;
      age?: number;
      sex?: string;
      height_cm?: number;
      weight_kg?: number;
      bmi?: number;
    },
    onProgress?: (p: number) => void
  ): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append("file", file);
    if (options.report_text) formData.append("report_text", options.report_text);
    formData.append("modality", options.modality || "T2");
    if (options.age) formData.append("age", String(options.age));
    if (options.sex) formData.append("sex", options.sex);
    if (options.height_cm) formData.append("height_cm", String(options.height_cm));
    if (options.weight_kg) formData.append("weight_kg", String(options.weight_kg));
    if (options.bmi) formData.append("bmi", String(options.bmi));

    const res = await this.client.post<PredictionResponse>("/predict/upload-mri", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (evt) => {
        if (onProgress && evt.total) {
          onProgress(Math.round((evt.loaded / evt.total) * 100));
        }
      },
    });
    return res.data;
  }

  // ── Reports ───────────────────────────────────────────────────────

  async getReportByStudy(studyId: string) {
    const res = await this.client.get(`/reports/study/${studyId}`);
    return res.data;
  }

  async downloadReportPDF(reportId: string): Promise<Blob> {
    const res = await this.client.get(`/reports/download/${reportId}/pdf`, {
      responseType: "blob",
    });
    return res.data;
  }

  // ── Analytics ─────────────────────────────────────────────────────

  async getAnalyticsSummary(): Promise<AnalyticsSummary> {
    const res = await this.client.get<AnalyticsSummary>("/analytics/summary");
    return res.data;
  }

  async getModelPerformance() {
    const res = await this.client.get("/analytics/model-performance");
    return res.data;
  }

  // ── Health ────────────────────────────────────────────────────────

  async healthCheck() {
    const res = await this.client.get("/health", { baseURL: "http://localhost:8000" });
    return res.data;
  }
}

export const api = new APIClient();
export default api;
