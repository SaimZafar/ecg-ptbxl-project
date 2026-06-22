const API = import.meta.env.VITE_API_URL || "https://saim0112-ecg-scope-api.hf.space";
export async function predictUpload(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/predict-upload`, { method: "POST", body: fd });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.detail || "Prediction failed");
  }
  return r.json();
}

export async function getSamples() {
  const r = await fetch(`${API}/samples`);
  if (!r.ok) throw new Error("Could not reach backend (is it running on :8000?)");
  return r.json();
}

export async function analyzeSample(index) {
  const r = await fetch(`${API}/analyze/${index}`);
  if (!r.ok) throw new Error("Analysis failed");
  return r.json();
}