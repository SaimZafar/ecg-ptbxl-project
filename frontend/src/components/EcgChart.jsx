export default function EcgChart({ trace, importance }) {
  const W = 1000, H = 300, mid = H / 2;
  const maxA = Math.max(...trace.map(Math.abs)) || 1;
  const scale = (H / 2 - 20) / maxA;
  const points = trace
    .map((v, i) => `${(i / (trace.length - 1)) * W},${mid - v * scale}`)
    .join(" ");
  const step = W / importance.length;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="ecg-svg" preserveAspectRatio="none">
      <g className="overlay">
        {importance.map((imp, i) =>
          imp > 0.05 ? (
            <rect key={i} x={i * step} y={0} width={step + 0.6} height={H}
              fill="var(--cam)" opacity={imp * 0.5} />
          ) : null
        )}
      </g>
      <polyline className="trace" points={points} pathLength={1}
        fill="none" stroke="var(--trace)" strokeWidth="1.5" />
    </svg>
  );
}