import { useEffect, useState } from "react";
import { getSamples, analyzeSample } from "./api";
import EcgChart from "./components/EcgChart";
import Prediction from "./components/Prediction";

export default function App() {
  const [samples, setSamples] = useState([]);
  const [selected, setSelected] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getSamples().then(setSamples).catch((e) => setError(e.message));
  }, []);

  async function run(index) {
    setSelected(index); setLoading(true); setError(null); setResult(null);
    try {
      setResult(await analyzeSample(index));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>ECG<span className="accent">·</span>Scope</h1>
        <p className="subtitle">Interpretable 12-lead ECG diagnosis · 1D-CNN + Grad-CAM</p>
      </header>

      <div className="picker">
        {samples.map((s) => (
          <button key={s.index}
            className={"chip" + (selected === s.index ? " active" : "")}
            onClick={() => run(s.index)}>
            {s.name}
            <span className="chip-label">{s.true_label}</span>
          </button>
        ))}
      </div>

      {error && <div className="error">⚠ {error} — is the backend running on :8000?</div>}
      {loading && <div className="status">Analyzing signal…</div>}

      {result && (
        <div className="results">
          <EcgChart key={selected} trace={result.trace} importance={result.importance} />
          <Prediction result={result} />
          <p className="legend">
            <span className="dot trace-dot"></span>Lead II trace
            <span className="dot cam-dot"></span>Model attention (Grad-CAM)
          </p>
        </div>
      )}
      {!result && !loading && !error && (
        <div className="status">Select a sample ECG to analyze.</div>
      )}
    </div>
  );
}