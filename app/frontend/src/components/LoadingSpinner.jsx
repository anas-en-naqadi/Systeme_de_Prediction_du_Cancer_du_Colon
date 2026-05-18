import { motion } from 'framer-motion';

export default function LoadingSpinner({ label = 'Loading' }) {
  return (
    <div className="loading-spinner" role="status" aria-live="polite" aria-label={label}>
      <motion.div
        className="spinner-ring"
        animate={{ rotate: 360 }}
        transition={{ duration: 1.2, repeat: Infinity, ease: 'linear' }}
      />
      <span>{label}...</span>
    </div>
  );
}
