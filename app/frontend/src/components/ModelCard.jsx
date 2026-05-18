import { useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

function formatPercent(value) {
  if (!Number.isFinite(Number(value))) {
    return '—';
  }

  const numeric = Number(value);
  if (numeric >= 0 && numeric <= 1) {
    return `${(numeric * 100).toFixed(1)}%`;
  }
  if (numeric > 1 && numeric <= 100) {
    return `${numeric.toFixed(1)}%`;
  }
  return numeric.toFixed(3);
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

export default function ModelCard({ metadata }) {
  const [expanded, setExpanded] = useState(false);

  const params = useMemo(() => {
    if (!metadata?.best_model_params || typeof metadata.best_model_params !== 'object') {
      return [];
    }
    return Object.entries(metadata.best_model_params);
  }, [metadata]);

  const ranking = useMemo(() => {
    if (!Array.isArray(metadata?.ranking_progression)) {
      return [];
    }
    return metadata.ranking_progression.slice(0, 6);
  }, [metadata]);

  return (
    <section className="model-card panel" aria-label="Model Card">
      <div className="model-card-head">
        <div>
          <span className="section-kicker">Model card</span>
          <h3>{metadata?.best_model || 'Model details'}</h3>
        </div>
        <button type="button" className="secondary-button model-card-toggle" onClick={() => setExpanded((current) => !current)}>
          {expanded ? 'Collapse' : 'Expand'}
        </button>
      </div>

      <div className="model-card-summary">
        <span><strong>Genes:</strong> {metadata?.selected_genes_count ?? 0}</span>
        <span><strong>Date:</strong> {formatDate(metadata?.training_date)}</span>
      </div>

      <AnimatePresence initial={false}>
        {expanded ? (
          <motion.div
            className="model-card-details"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22 }}
          >
            <div className="model-card-grid">
              <div className="model-card-block">
                <h4>Evaluation</h4>
                <p><strong>Method:</strong> {metadata?.evaluation_method || '—'}</p>
                <p><strong>Training set:</strong> {metadata?.training_set || '—'}</p>
                <p><strong>ROC-AUC:</strong> {formatPercent(metadata?.roc_auc)}</p>
                <p><strong>F1-score:</strong> {formatPercent(metadata?.f1_score)}</p>
              </div>

              <div className="model-card-block">
                <h4>Hyperparameters</h4>
                {params.length === 0 ? (
                  <p>—</p>
                ) : (
                  params.map(([key, value]) => (
                    <p key={key}>
                      <strong>{key}:</strong> {String(value)}
                    </p>
                  ))
                )}
              </div>

              <div className="model-card-block">
                <h4>Selected genes</h4>
                <div className="tag-row">
                  {(metadata?.selected_genes || []).map((gene) => (
                    <span key={gene} className="badge-chip gene-tag">{gene}</span>
                  ))}
                </div>
              </div>

              <div className="model-card-block">
                <h4>Ranking progression</h4>
                <div className="rank-list">
                  {ranking.length === 0 ? (
                    <p>—</p>
                  ) : (
                    ranking.map((item) => (
                      <div key={`${item.step}-${item.feature}`} className="rank-item">
                        <span>#{item.step} {item.feature}</span>
                        <strong>{formatPercent(item.cv_mean_auc)}</strong>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </section>
  );
}
