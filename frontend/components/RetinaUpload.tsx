"use client";

import { useState } from "react";

interface PredictionResult {
  prediction: string;
  confidence: number;
  all_probabilities: Record<string, number>;
}

const severityInfo: Record<string, { color: string; bg: string; description: string; label: string }> = {
  "No DR": {
    color: "var(--sev-none)", bg: "#e8f7f0",
    label: "No DR Detected",
    description: "No visible signs of diabetic retinopathy in this image.",
  },
  "Mild": {
    color: "var(--sev-mild)", bg: "#fbf1dc",
    label: "Mild NPDR",
    description: "Early-stage changes present. Routine annual follow-up recommended.",
  },
  "Moderate": {
    color: "var(--sev-moderate)", bg: "#fdeee0",
    label: "Moderate NPDR",
    description: "Noticeable vascular changes present. Follow-up within 6-12 months advised.",
  },
  "Severe": {
    color: "var(--sev-severe)", bg: "#fbe6e0",
    label: "Severe NPDR",
    description: "Significant changes present. Prompt specialist evaluation recommended.",
  },
  "Proliferative": {
    color: "var(--sev-proliferative)", bg: "#f7ded9",
    label: "Proliferative DR",
    description: "Advanced-stage changes present. Urgent specialist evaluation strongly recommended.",
  },
};

export default function RetinaUpload() {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setPreviewUrl(URL.createObjectURL(file));
    setUploading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Prediction failed");
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Try again.");
    } finally {
      setUploading(false);
    }
  };

  const info = result ? severityInfo[result.prediction] : null;

  return (
    <div>
      {/* Scan upload zone */}
      <label
        className="scan-zone relative block border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all"
        style={{ borderColor: "var(--clinic-teal)" }}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          disabled={uploading}
          className="hidden"
        />

        {previewUrl ? (
          <img
            src={previewUrl}
            alt="Uploaded scan"
            className={`w-32 h-32 object-cover rounded-full mx-auto border-4 ${uploading ? "pulse-ring" : ""}`}
            style={{ borderColor: "var(--clinic-teal)" }}
          />
        ) : (
          <div
            className="w-16 h-16 rounded-full mx-auto flex items-center justify-center mb-3"
            style={{ backgroundColor: "var(--clinic-teal-light)" }}
          >
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="3.5" stroke="var(--clinic-teal)" strokeWidth="1.6" />
              <path
                d="M2 12c2.5-5 6.5-8 10-8s7.5 3 10 8c-2.5 5-6.5 8-10 8s-7.5-3-10-8z"
                stroke="var(--clinic-teal)" strokeWidth="1.6" fill="none"
              />
            </svg>
          </div>
        )}

        <p className="mt-4 font-semibold" style={{ color: "var(--clinic-navy)" }}>
          {uploading ? "Analyzing scan…" : previewUrl ? "Upload a different scan" : "Click to upload retina scan"}
        </p>
        <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
          JPG or PNG fundus photograph
        </p>
      </label>

      {error && (
        <p className="mt-4 text-sm font-medium" style={{ color: "var(--sev-severe)" }}>
          {error}
        </p>
      )}

      {/* Clinical result report */}
      {result && info && (
        <div className="mt-8 rounded-2xl overflow-hidden border" style={{ borderColor: "var(--border)" }}>
          <div className="px-6 py-4 flex items-center justify-between" style={{ backgroundColor: info.bg }}>
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                Screening Result
              </p>
              <p className="text-xl font-bold" style={{ color: info.color }}>
                {info.label}
              </p>
            </div>
            <div
              className="w-16 h-16 rounded-full flex items-center justify-center font-mono font-bold text-sm"
              style={{ backgroundColor: "#fff", border: `3px solid ${info.color}`, color: info.color }}
            >
              {result.confidence}%
            </div>
          </div>

          <div className="px-6 py-5" style={{ backgroundColor: "var(--card)" }}>
            <p className="text-sm leading-relaxed mb-5" style={{ color: "var(--text)" }}>
              {info.description}
            </p>

            <p className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: "var(--text-muted)" }}>
              Full Probability Breakdown
            </p>
            <div className="space-y-2.5">
              {Object.entries(result.all_probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([className, prob]) => (
                  <div key={className} className="flex items-center gap-3 text-sm">
                    <span className="w-28 font-medium" style={{ color: "var(--text)" }}>{className}</span>
                    <div className="flex-1 rounded-full h-2 overflow-hidden" style={{ backgroundColor: "var(--bg)" }}>
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${prob}%`,
                          backgroundColor: severityInfo[className]?.color || "#999",
                        }}
                      />
                    </div>
                    <span className="w-12 text-right font-mono" style={{ color: "var(--text-muted)" }}>
                      {prob}%
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}