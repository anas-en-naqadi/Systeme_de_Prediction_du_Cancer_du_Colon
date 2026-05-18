import './HeaderGraphic.css';

export default function HeaderGraphic({ health }) {
  const loaded = !!health?.model_loaded;

  return (
    <div className="header-graphic" aria-hidden="true">
      <svg className="dna-svg" viewBox="0 0 220 80" preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id="g" x1="0" x2="1">
            <stop offset="0%" stopColor="#8b5cf6" />
            <stop offset="50%" stopColor="#06b6d4" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
        </defs>

        <path className="dna-path" d="M10 70 C40 10, 80 10, 110 70 C140 130, 180 130, 210 70" stroke="url(#g)" fill="none" />
        <path className="dna-path dna-path-2" d="M10 10 C40 70, 80 70, 110 10 C140 -50, 180 -50, 210 10" stroke="url(#g)" fill="none" />

        <g className="helix-dots">
          <circle className="dot" cx="40" cy="36" r="3" />
          <circle className="dot" cx="80" cy="36" r="3" style={{ animationDelay: '0.3s' }} />
          <circle className="dot" cx="120" cy="36" r="3" style={{ animationDelay: '0.6s' }} />
          <circle className="dot" cx="160" cy="36" r="3" style={{ animationDelay: '0.9s' }} />
        </g>

        <rect x="186" y="8" width="28" height="28" rx="6" className={loaded ? 'status-box ok' : 'status-box off'} />
      </svg>
    </div>
  );
}
