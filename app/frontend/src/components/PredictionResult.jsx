import { motion, AnimatePresence } from 'framer-motion';
import ConfidenceBar from './ConfidenceBar';

export default function PredictionResult({ result, loading }) {
  const predictionTone = result?.prediction === 'Abnormal' ? 'abnormal' : 'normal';
  const isBorderline = result && result.confidence >= 50 && result.confidence <= 65;

  return (
    <section className="prediction-shell" aria-live="polite" aria-atomic="true">
      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div
            key="prediction-loading"
            className="prediction-card prediction-skeleton"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="skeleton-block large" />
            <div className="skeleton-block medium" />
            <div className="skeleton-block small" />
          </motion.div>
        ) : result ? (
          <motion.div
            key="prediction-result"
            className={`prediction-card ${predictionTone} ${isBorderline ? 'borderline' : ''}`}
            initial={{ opacity: 0, y: 24, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.45, ease: 'easeOut' }}
          >
            <div className="result-eyebrow">Inference result</div>
            <h2>{result.prediction}</h2>
            <p className="result-confidence">{result.confidence.toFixed(1)}% confidence</p>

            {isBorderline && (
              <motion.div
                className="borderline-warning"
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <span className="warning-icon">⚠️</span>
                <span className="warning-text">Borderline result — specialist review recommended</span>
              </motion.div>
            )}

            <div className="probability-grid">
              <div>
                <span>Normal</span>
                <strong>{(result.probabilities.normal * 100).toFixed(1)}%</strong>
              </div>
              <div>
                <span>Abnormal</span>
                <strong>{(result.probabilities.abnormal * 100).toFixed(1)}%</strong>
              </div>
            </div>

            <ConfidenceBar
              confidence={result.confidence}
              variant={predictionTone}
            />

            <div className="result-disclaimer">
              <strong>Clinical Notice:</strong> This is a decision support tool only. Do not use as the sole diagnostic criterion. Always consult medical professionals for clinical decisions.
            </div>

            <div className="result-footnote">
              Model: <strong>{result.model}</strong>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="prediction-empty"
            className="prediction-card prediction-empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <div className="empty-orb" />
            <h2>Awaiting analysis</h2>
            <p>Adjust the selected genes and run the clinical inference workflow.</p>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
