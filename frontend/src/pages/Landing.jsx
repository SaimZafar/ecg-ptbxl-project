import { Link } from "react-router-dom";
import EcgLine from "../components/EcgLine";

export default function Landing() {
  return (
    <div className="landing">
      <section className="page hero">
        <div className="eyebrow">Research preview · Explainable AI</div>
        <h1 className="hero-title">Read an ECG.<br />See the reasoning.</h1>
        <p className="lede">Upload a 12-lead ECG and get an AI-assisted reading across five
          diagnostic categories — with a visual map of exactly where the model looked.</p>
        <div className="cta-row">
          <Link to="/signup" className="btn primary">Get started</Link>
          <Link to="/login" className="btn ghost">Log in</Link>
        </div>
      </section>

      <div className="ecg-band"><EcgLine width={680} height={120} beats={7} color="var(--ink)" /></div>

      <div className="stat-row">
        <div className="stat"><div className="stat-num">5</div><div className="stat-label">Diagnostic classes</div></div>
        <div className="stat"><div className="stat-num">30k+</div><div className="stat-label">ECGs trained on</div></div>
        <div className="stat"><div className="stat-num">Grad-CAM</div><div className="stat-label">Explanations</div></div>
      </div>

      <section className="page how">
        <h2>How it works</h2>
        <ol>
          <li>Create an account and log in.</li>
          <li>Upload a 12-lead, 10-second ECG file.</li>
          <li>See predicted categories, confidence, and a Grad-CAM overlay on the trace.</li>
        </ol>
        <p className="muted">Built on a 1D-CNN trained on PTB-XL + Chapman-Shaoxing, with test-time
          adaptation for robustness across recording sources.</p>
      </section>

      <p className="disclaimer page-disclaimer">Research &amp; educational tool only.
        <strong> Not for clinical use</strong> and not a medical device. Do not upload real,
        identifiable patient data.</p>
    </div>
  );
}