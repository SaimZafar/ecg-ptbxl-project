function ecgPoints(w, h, beats) {
  const mid = h * 0.52, pts = [], per = w / beats;
  for (let x = 0; x <= w; x++) {
    const t = (x % per) / per; let y = 0;
    if (t < 0.12) y = Math.sin((t / 0.12) * Math.PI) * 0.10;
    else if (t >= 0.16 && t < 0.20) y = -((t - 0.16) / 0.04) * 0.12;
    else if (t >= 0.20 && t < 0.24) y = -0.12 + ((t - 0.20) / 0.04) * 1.02;
    else if (t >= 0.24 && t < 0.28) y = 0.90 - ((t - 0.24) / 0.04) * 1.02;
    else if (t >= 0.28 && t < 0.32) y = -0.12 + ((t - 0.28) / 0.04) * 0.12;
    else if (t >= 0.42 && t < 0.60) y = Math.sin(((t - 0.42) / 0.18) * Math.PI) * 0.20;
    pts.push(`${x},${(mid - y * h * 0.72).toFixed(1)}`);
  }
  return pts.join(" ");
}

export default function EcgLine({ width = 680, height = 120, beats = 7, color = "var(--ink)" }) {
  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none"
      style={{ display: "block", width: "100%", height }}>
      <polyline points={ecgPoints(width, height, beats)} fill="none" stroke={color}
        strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" className="ecg-anim" />
    </svg>
  );
}