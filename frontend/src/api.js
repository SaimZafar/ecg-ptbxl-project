const API = "http://127.0.0.1:8000";

export async function getSamples() {
  const r = await fetch(`${API}/samples`);
  if (!r.ok) throw new Error("Failed to load samples");
  return r.json();
}

export async function analyzeSample(index) {
  const r = await fetch(`${API}/analyze/${index}`);
  if (!r.ok) throw new Error("Analysis failed");
  return r.json();
}