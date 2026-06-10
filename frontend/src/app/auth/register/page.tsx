"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { toast } from "react-toastify";
import api from "@/lib/api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const schema = z.object({
  email: z.string().email("Invalid email"),
  username: z.string().min(3, "Min 3 characters").max(50),
  full_name: z.string().optional(),
  password: z.string().min(8, "Min 8 characters"),
  confirm_password: z.string(),
}).refine(d => d.password === d.confirm_password, {
  message: "Passwords don't match",
  path: ["confirm_password"],
});
type FormData = z.infer<typeof schema>;

export default function RegisterPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const { register, handleSubmit, formState: { errors } } = useForm<FormData>({
    resolver: zodResolver(schema),
  });

  const onSubmit = async (data: FormData) => {
    setLoading(true);
    try {
      await api.register({
        email: data.email,
        username: data.username,
        full_name: data.full_name,
        password: data.password,
      });
      toast.success("Account created! Please sign in.");
      router.push("/auth/login");
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  const fields: Array<{ name: keyof FormData; label: string; type?: string; placeholder?: string }> = [
    { name: "full_name",        label: "Full Name",        type: "text",     placeholder: "Dr. Jane Smith" },
    { name: "username",         label: "Username",         type: "text",     placeholder: "janesmith" },
    { name: "email",            label: "Email",            type: "email",    placeholder: "jane@hospital.com" },
    { name: "password",         label: "Password",         type: "password", placeholder: "Min 8 characters" },
    { name: "confirm_password", label: "Confirm Password", type: "password", placeholder: "Repeat password" },
  ];

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-blue-600 mb-4">
            <svg className="w-9 h-9 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
            </svg>
          </div>
          <h1 className="text-3xl font-bold text-white">Create Account</h1>
          <p className="text-blue-300 mt-1 text-sm">Join the ATM-Net++ platform</p>
        </div>

        <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl p-8 shadow-2xl">
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            {fields.map(f => (
              <div key={f.name}>
                <label className="block text-sm font-medium text-blue-200 mb-1">{f.label}</label>
                <input
                  {...register(f.name)}
                  type={f.type || "text"}
                  placeholder={f.placeholder}
                  className={cn(
                    "w-full px-4 py-2.5 rounded-lg bg-white/10 border text-white placeholder-white/40",
                    "focus:outline-none focus:ring-2 focus:ring-blue-500 transition",
                    errors[f.name] ? "border-red-400" : "border-white/20"
                  )}
                />
                {errors[f.name] && (
                  <p className="text-red-400 text-xs mt-1">{errors[f.name]?.message as string}</p>
                )}
              </div>
            ))}
            <Button type="submit" loading={loading} className="w-full h-11 mt-2">
              Create Account
            </Button>
          </form>
          <p className="text-center text-white/50 text-xs mt-6">
            Already have an account?{" "}
            <a href="/auth/login" className="text-blue-400 hover:text-blue-300">Sign in</a>
          </p>
        </div>
      </div>
    </div>
  );
}
