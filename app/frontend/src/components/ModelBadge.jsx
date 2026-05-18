import { motion } from 'framer-motion';

function MetricBlock({ label, value }) {
  const formatMetric = (v) => {
    if (v == null) return '—';
    if (typeof v === 'number') {
      // If metric is a proportion (0..1), render as percentage with one decimal
      if (v >= 0 && v <= 1) return `${(v * 100).toFixed(1)}%`;
      // If metric is already in 0..100 range, render with one decimal and %
      if (v > 1 && v <= 100) return `${v.toFixed(1)}%`;
      // Otherwise fallback to 3-decimal numeric
      return v.toFixed(3);
    }
    return String(v);
  };

  return (
    <div className="model-metric-block">
      <span>{label}</span>
      <strong>{formatMetric(value)}</strong>
    </div>
  );
}

export default function ModelBadge({ metadata }) {
  if (!metadata) {
    return null;
  }

  return (
    <motion.section
      className="model-badge"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15, duration: 0.4 }}
    >
      <div className="model-badge-head">
        <div>
          <span className="section-kicker">Model intelligence</span>
          <h2>{metadata.best_model}</h2>
        </div>
        <span className="badge-chip">{metadata.selected_genes_count} genes</span>
      </div>

      <div className="model-grid">
        <MetricBlock label="ROC-AUC" value={metadata.roc_auc} />
        <MetricBlock label="Accuracy" value={metadata.accuracy} />
        <MetricBlock label="Precision" value={metadata.precision} />
        <MetricBlock label="F1-score" value={metadata.f1_score} />
        <MetricBlock label="Recall" value={metadata.recall} />
      </div>
    </motion.section>
  );
}
