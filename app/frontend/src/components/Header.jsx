import { motion } from 'framer-motion';
import HeaderGraphic from './HeaderGraphic';

export default function Header({ health, loading }) {
  const statusLabel = loading ? 'Connecting' : health?.status || 'Offline';
  const statusClass = loading ? 'status-warn' : health?.model_loaded ? 'status-ok' : 'status-error';

  return (
    <motion.header
      className="hero-header"
      initial={{ opacity: 0, y: -18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55 }}
    >
      <div>
        <div className="brand-line">
          <span className="brand-pill">Clinical AI</span>
          <span className={`status-dot ${statusClass}`} />
          <span className="status-text">{statusLabel}</span>
        </div>
        <h1>Colon Cancer Prediction Dashboard</h1>
        <p>
          Premium biomedical inference interface for gene-expression based colon cancer risk assessment.
        </p>
      </div>

      <HeaderGraphic health={health} />
    </motion.header>
  );
}
