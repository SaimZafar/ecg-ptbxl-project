import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { supabase } from "../supabaseClient";
import EcgLine from "../components/EcgLine";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const nav = useNavigate();

  async function submit(e) {
    e.preventDefault(); setBusy(true); setError(null);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    setBusy(false);
    if (error) setError(error.message); else nav("/analyze");
  }

  return (
    <div className="auth-split">
      <aside className="auth-aside">
        <Link to="/" className="brand light">ECG<span className="accent">·</span>Scope</Link>
        <EcgLine width={240} height={56} beats={4} color="var(--bg)" />
        <p className="aside-quote">Every beat, explained.</p>
      </aside>
      <div className="auth-main">
        <h1>Welcome back</h1>
        <p className="auth-sub">Log in to analyze an ECG</p>
        <form onSubmit={submit} className="form">
          <label className="field-label">Email</label>
          <input type="email" placeholder="you@example.com" value={email}
            onChange={(e) => setEmail(e.target.value)} required />
          <label className="field-label">Password</label>
          <input type="password" placeholder="••••••••" value={password}
            onChange={(e) => setPassword(e.target.value)} required />
          {error && <div className="error">{error}</div>}
          <button className="btn primary full" disabled={busy}>{busy ? "…" : "Log in"}</button>
        </form>
        <p className="muted">No account? <Link to="/signup">Sign up</Link></p>
      </div>
    </div>
  );
}