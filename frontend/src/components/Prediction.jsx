export default function Prediction({ result }) {
  const entries = Object.entries(result.probabilities);
  return (
    <div className="prediction">
      <div className="verdict">
        <span className="verdict-label">Diagnosis</span>
        <span className="verdict-class">{result.predicted_class}</span>
        <span className="verdict-true">actual: {result.true_label}</span>
      </div>
      <div className="bars">
        {entries.map(([cls, p]) => (
          <div className="bar-row" key={cls}>
            <span className="bar-label">{cls}</span>
            <div className="bar-track">
              <div className={"bar-fill" + (cls === result.predicted_class ? " top" : "")}
                style={{ width: `${(p * 100).toFixed(1)}%` }} />
            </div>
            <span className="bar-val">{(p * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}