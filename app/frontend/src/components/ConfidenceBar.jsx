import { motion } from 'framer-motion';

export default function ConfidenceBar({ confidence = 0, variant = 'neutral' }) {
  const fillClass = variant === 'abnormal' ? 'fill-danger' : variant === 'normal' ? 'fill-success' : 'fill-neutral';

  return (
    <div className="confidence-wrapper">
      <div className="confidence-head">
        <span>Confidence</span>
        <strong>{confidence.toFixed(1)}%</strong>
      </div>

      <div className="confidence-track" aria-label="Prediction confidence bar">
        <motion.div
          className={`confidence-fill ${fillClass}`}
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(Math.max(confidence, 0), 100)}%` }}
          transition={{ duration: 0.9, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
}
