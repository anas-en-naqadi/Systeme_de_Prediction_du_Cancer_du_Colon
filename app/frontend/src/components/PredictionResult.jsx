import { motion, AnimatePresence } from 'framer-motion';
import ConfidenceBar from './ConfidenceBar';

export default function PredictionResult({ result, loading }) {
  const predictionTone = result?.prediction === 'Abnormal' ? 'abnormal' : 'normal';

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
            className={`prediction-card ${predictionTone}`}
            initial={{ opacity: 0, y: 24, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.45, ease: 'easeOut' }}
          >
            <div className="result-eyebrow">Inference result</div>
            <h2>{result.prediction}</h2>
            <p className="result-confidence">{result.confidence.toFixed(1)}% confidence</p>

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
