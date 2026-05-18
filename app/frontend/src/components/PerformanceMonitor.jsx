import { motion } from 'framer-motion';

function formatUptime(seconds) {
  const value = Number.isFinite(Number(seconds)) ? Math.max(0, Math.floor(Number(seconds))) : 0;
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const remaining = value % 60;
  return [hours, minutes, remaining].map((part) => String(part).padStart(2, '0')).join(':');
}

function formatNumber(value, digits = 2) {
  if (!Number.isFinite(Number(value))) {
    return '—';
  }
  return Number(value).toFixed(digits);
}

function Kpi({ label, value }) {
  return (
    <div className="perf-kpi">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function PerformanceMonitor({ metrics, stale = false }) {
  return (
    <motion.section
      className="performance-monitor"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2, duration: 0.35 }}
      aria-live="polite"
      aria-label="Performance Monitor"
    >
      <div className="performance-head">
        <div>
          <span className="section-kicker">Performance monitor</span>
          <h3>Runtime telemetry</h3>
        </div>
        <span className={`badge-chip ${stale ? 'badge-chip-warn' : ''}`}>
          {stale ? 'Stale feed' : 'Live'}
        </span>
      </div>

      <div className="performance-grid">
        <Kpi label="Average latency" value={`${formatNumber(metrics?.average_latency_ms, 1)} ms`} />
        <Kpi label="Throughput (60s)" value={`${formatNumber(metrics?.throughput_rps_60s, 3)} rps`} />
        <Kpi label="Uptime" value={formatUptime(metrics?.uptime_seconds)} />
        <Kpi label="Total requests" value={Number.isFinite(Number(metrics?.total_requests)) ? String(metrics.total_requests) : '—'} />
      </div>
    </motion.section>
  );
}
