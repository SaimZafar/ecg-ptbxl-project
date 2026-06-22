import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { supabase } from "../supabaseClient";
import EcgLine from "../components/EcgLine";

export default function Signup() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const nav = useNavigate();

  async function submit(e) {
    e.preventDefault(); setBusy(true); setError(null); setMsg(null);
    const { data, error } = await supabase.auth.signUp({ email, password });
    setBusy(false);
    if (error) return setError(error.message);
    data.session ? nav("/analyze") : setMsg("Check your email to confirm, then log in.");
  }

  return (
    <div className="auth-split">
      <aside className="auth-aside">
        <Link to="/" className="brand light">ECG<span className="accent">·</span>Scope</Link>
        <EcgLine width={240} height={56} beats={4} color="var(--bg)" />
        <p className="aside-quote">Every beat, explained.</p>
      </aside>
      <div className="auth-main">
        <h1>Create account</h1>
        <p className="auth-sub">Start analyzing ECGs in seconds</p>
        <form onSubmit={submit} className="form">
          <label className="field-label">Email</label>
          <input type="email" placeholder="you@example.com" value={email}
            onChange={(e) => setEmail(e.target.value)} required />
          <label className="field-label">Password</label>
          <input type="password" placeholder="Min 6 characters" value={password}
            onChange={(e) => setPassword(e.target.value)} required minLength={6} />
          {error && <div className="error">{error}</div>}
          {msg && <div className="status">{msg}</div>}
          <button className="btn primary full" disabled={busy}>{busy ? "…" : "Sign up"}</button>
        </form>
        <p className="muted">Have an account? <Link to="/login">Log in</Link></p>
      </div>
    </div>
  );
}