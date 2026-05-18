import { motion } from 'framer-motion';

export default function GeneInputCard({ gene, value, onChange, index }) {
  const sliderMin = 0;
  const sliderMax = 10000;

  const handleInputChange = (event) => {
    const raw = event.target.value;
    if (raw === '') {
      onChange('');
      return;
    }

    const parsed = Number(raw);
    if (!Number.isNaN(parsed)) {
      onChange(parsed);
    }
  };

  const handleSliderChange = (event) => {
    onChange(Number(event.target.value));
  };

  return (
    <motion.article
      className="gene-card"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08, duration: 0.35 }}
      whileHover={{ y: -4, scale: 1.01 }}
    >
      <div className="gene-card-head">
        <div>
          <span className="gene-label">Gene marker</span>
          <h3>{gene}</h3>
        </div>
        <span className="gene-badge">Expression</span>
      </div>

      <div className="gene-info">
        <p>
          Adjust the measured expression level for <strong>{gene}</strong>. Numeric values are required.
        </p>
      </div>

      <div className="gene-input-row">
        <label htmlFor={`gene-input-${gene}`} className="sr-only">
          {gene} expression value
        </label>
        <input
          id={`gene-input-${gene}`}
          className="gene-number-input"
          type="number"
          inputMode="decimal"
          step="0.01"
          min={sliderMin}
          max={sliderMax}
          value={value}
          onChange={handleInputChange}
          aria-label={`${gene} expression value`}
        />
      </div>

      <input
        className="gene-slider"
        type="range"
        min={sliderMin}
        max={sliderMax}
        step="1"
        value={Number.isFinite(value) ? value : 0}
        onChange={handleSliderChange}
        aria-label={`${gene} expression slider`}
      />

      <div className="gene-scale">
        <span>0</span>
        <span>5k</span>
        <span>10k</span>
      </div>
    </motion.article>
  );
}
