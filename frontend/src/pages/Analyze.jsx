import { useState, useEffect, useCallback } from "react";
import { useAuth } from "../auth";
import { supabase } from "../supabaseClient";
import { predictUpload, getSamples, analyzeSample } from "../api";
import EcgChart from "../components/EcgChart";
import Prediction from "../components/Prediction";

export default function Analyze() {
  const { user } = useAuth();
  const [samples, setSamples] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [history, setHistory] = useState([]);

  const loadHistory = useCallback(async () => {
    const { data, error } = await supabase.from("analyses")
      .select("id, created_at, source, predicted_class")
      .order("created_at", { ascending: false }).limit(10);
    if (error) console.error("history load failed:", error);
    setHistory(data || []);
  }, []);

  useEffect(() => { getSamples().then(setSamples).catch(() => {}); loadHistory(); }, [loadHistory]);

  async function run(promise, source) {
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await promise;
      setResult(r);
      const { error: insErr } = await supabase.from("analyses").insert({
        user_id: user.id, source,
        predicted_class: r.predicted_class, probabilities: r.probabilities,
      });
      if (insErr) console.error("history insert failed:", insErr);
      loadHistory();
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }

  function onFile(e) {
    const file = e.target.files[0]; if (!file) return;
    setFileName(file.name); run(predictUpload(file), file.name);
  }
  function trySample(s) { setFileName(null); run(analyzeSample(s.index), `sample: ${s.true_label}`); }

  return (
    <div className="page">
      <div className="analyze-head">
        <h1>Analyze an ECG</h1>
        <p className="muted">Signed in as {user?.email}</p>
      </div>

      <label className="upload-drop">
        <input type="file" accept=".csv,.npy,.txt" onChange={onFile} hidden />
        <span className="upload-icon">⌁</span>
        <span className="upload-text">{fileName || "Choose a 12-lead ECG file (.csv or .npy)"}</span>
        <span className="upload-hint">12 leads × ~10 seconds — resampled automatically</span>
      </label>

      {samples.length > 0 && (
        <div className="try-row">
          <span className="try-label">Or try a sample:</span>
          {samples.map((s) => (
            <button key={s.index} className="chip" onClick={() => trySample(s)}>{s.true_label}</button>
          ))}
        </div>
      )}

      {error && <div className="error">⚠ {error}</div>}
      {loading && <div className="status">Analyzing…</div>}

      {result && (
        <div className="results">
          <EcgChart key={fileName || "sample"} trace={result.trace} importance={result.importance} />
          <Prediction result={result} />
          <p className="legend"><span className="dot trace-dot"></span>Lead II trace
            <span className="dot cam-dot"></span>Model attention</p>
          <p className="disclaimer">{result.disclaimer}</p>
        </div>
      )}

      {history.length > 0 && (
        <div className="history">
          <h2>Recent analyses</h2>
          <ul className="history-list">
            {history.map((h) => (
              <li key={h.id}>
                <span className="hist-class">{h.predicted_class}</span>
                <span className="hist-src">{h.source}</span>
                <span className="hist-date">{new Date(h.created_at).toLocaleString()}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}