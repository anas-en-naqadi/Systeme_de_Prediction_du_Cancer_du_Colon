import { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import Header from './components/Header';
import GeneInputCard from './components/GeneInputCard';
import PredictionResult from './components/PredictionResult';
import ModelBadge from './components/ModelBadge';
import ModelCard from './components/ModelCard';
import PerformanceMonitor from './components/PerformanceMonitor';
import LoadingSpinner from './components/LoadingSpinner';
import { getGenes, getHealth, getMetadata, getMetrics, predict } from './api/client';

const DEFAULT_GENE_VALUE = 2500;

function toToastMessage(error) {
  if (!error) {
    return 'Something went wrong.';
  }

  if (typeof error === 'string') {
    return error;
  }

  if (error.message) {
    return error.message;
  }

  return 'Unable to complete the request.';
}

function buildDefaultGeneState(genes) {
  return genes.reduce((accumulator, gene) => {
    accumulator[gene] = DEFAULT_GENE_VALUE;
    return accumulator;
  }, {});
}

function validateGeneValues(geneValues, selectedGenes) {
  const missing = selectedGenes.filter((gene) => geneValues[gene] === '' || geneValues[gene] === undefined || geneValues[gene] === null);
  const invalid = selectedGenes.filter((gene) => {
    const value = geneValues[gene];
    return value !== '' && (!Number.isFinite(Number(value)) || Number.isNaN(Number(value)));
  });

  return { missing, invalid };
}

export default function App() {
  const [genes, setGenes] = useState([]);
  const [metadata, setMetadata] = useState(null);
  const [health, setHealth] = useState(null);
  const [geneValues, setGeneValues] = useState({});
  const [result, setResult] = useState(null);
  const [loadingPage, setLoadingPage] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [toast, setToast] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [metricsStale, setMetricsStale] = useState(false);

  useEffect(() => {
    let isMounted = true;

    const loadDashboard = async () => {
      try {
        const [genesResponse, metadataResponse, healthResponse] = await Promise.all([
          getGenes(),
          getMetadata(),
          getHealth(),
        ]);

        if (!isMounted) {
          return;
        }

        const receivedGenes = genesResponse.genes || [];
        setGenes(receivedGenes);
        setMetadata(metadataResponse);
        setHealth(healthResponse);
        setGeneValues(buildDefaultGeneState(receivedGenes));
      } catch (error) {
        if (isMounted) {
          setToast({ type: 'error', message: toToastMessage(error) });
        }
      } finally {
        if (isMounted) {
          setLoadingPage(false);
        }
      }
    };

    loadDashboard();

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!toast) {
      return undefined;
    }

    const timeout = window.setTimeout(() => setToast(null), 4500);
    return () => window.clearTimeout(timeout);
  }, [toast]);

  useEffect(() => {
    let isMounted = true;

    const fetchMetrics = async () => {
      try {
        const payload = await getMetrics();
        if (!isMounted) {
          return;
        }
        setMetrics(payload);
        setMetricsStale(false);
      } catch {
        if (isMounted) {
          setMetricsStale(true);
        }
      }
    };

    fetchMetrics();
    const timerId = window.setInterval(fetchMetrics, 5000);

    return () => {
      isMounted = false;
      window.clearInterval(timerId);
    };
  }, []);

  const selectedAlgorithm = metadata?.best_model || health?.selected_algorithm || 'Model pending';

  const canAnalyze = useMemo(() => {
    if (loadingPage || predicting || genes.length === 0) {
      return false;
    }

    return genes.every((gene) => Number.isFinite(Number(geneValues[gene])));
  }, [genes, geneValues, loadingPage, predicting]);

  const handleGeneChange = (gene, nextValue) => {
    setGeneValues((current) => ({
      ...current,
      [gene]: nextValue,
    }));
  };

  const handleAnalyze = async () => {
    const { missing, invalid } = validateGeneValues(geneValues, genes);

    if (missing.length > 0 || invalid.length > 0) {
      setToast({
        type: 'error',
        message: missing.length > 0
          ? `Please complete all selected genes: ${missing.join(', ')}`
          : `Invalid numeric values detected: ${invalid.join(', ')}`,
      });
      return;
    }

    const payload = genes.reduce((accumulator, gene) => {
      accumulator[gene] = Number(geneValues[gene]);
      return accumulator;
    }, {});

    try {
      setPredicting(true);
      setResult(null);
      const response = await predict(payload);
      setResult(response);
    } catch (error) {
      setToast({ type: 'error', message: toToastMessage(error) });
    } finally {
      setPredicting(false);
    }
  };

  const handleReset = () => {
    setGeneValues(buildDefaultGeneState(genes));
    setResult(null);
    setToast(null);
  };

  return (
    <div className="app-shell">
      <div className="bg-orb orb-a" />
      <div className="bg-orb orb-b" />

      <main className="dashboard-page">
        <Header health={health} loading={loadingPage} />

        <motion.section
          className="dashboard-meta-row"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.4 }}
        >
          <div className="info-chip">Selected algorithm: <strong>{selectedAlgorithm}</strong></div>
          <div className="info-chip">Backend: <strong>{health?.status || 'loading'}</strong></div>
          <div className="info-chip">Genes loaded: <strong>{genes.length || 0}</strong></div>
        </motion.section>

        <PerformanceMonitor metrics={metrics} stale={metricsStale} />

        {loadingPage ? (
          <div className="loading-state-grid" aria-busy="true" aria-live="polite">
            <div className="loading-panel">
              <LoadingSpinner label="Loading genes" />
              <div className="skeleton-list">
                {Array.from({ length: 6 }).map((_, index) => (
                  <div className="skeleton-card" key={`skeleton-${index}`}>
                    <div className="skeleton-line long" />
                    <div className="skeleton-line short" />
                    <div className="skeleton-line medium" />
                  </div>
                ))}
              </div>
            </div>

            <div className="loading-panel loading-preview">
              <LoadingSpinner label="Loading metadata" />
              <div className="prediction-card prediction-skeleton">
                <div className="skeleton-block large" />
                <div className="skeleton-block medium" />
                <div className="skeleton-block small" />
              </div>
            </div>
          </div>
        ) : (
          <div className="dashboard-grid">
            <section className="panel panel-inputs" aria-label="gene inputs">
              <div className="panel-head">
                <div>
                  <span className="section-kicker">Gene panel</span>
                  <h2>Patient expression values</h2>
                </div>

                <div className="panel-actions">
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={handleReset}
                    disabled={predicting}
                  >
                    Reset
                  </button>
                  <button
                    type="button"
                    className="primary-button"
                    onClick={handleAnalyze}
                    disabled={!canAnalyze}
                  >
                    {predicting ? 'Analyzing...' : 'Analyze'}
                  </button>
                </div>
              </div>

              <div className="gene-grid">
                <AnimatePresence>
                  {genes.map((gene, index) => (
                    <GeneInputCard
                      key={gene}
                      gene={gene}
                      value={geneValues[gene] ?? ''}
                      index={index}
                      onChange={(nextValue) => handleGeneChange(gene, nextValue)}
                    />
                  ))}
                </AnimatePresence>
              </div>
            </section>

            <section className="panel panel-results" aria-label="prediction results">
              <PredictionResult result={result} loading={predicting} />
              <ModelBadge metadata={metadata} />
              <ModelCard metadata={metadata} />
            </section>
          </div>
        )}
      </main>

      <AnimatePresence>
        {toast ? (
          <motion.div
            className={`toast ${toast.type}`}
            initial={{ opacity: 0, y: 16, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12 }}
            role="status"
            aria-live="polite"
          >
            <strong>{toast.type === 'error' ? 'Request failed' : 'Notice'}</strong>
            <span>{toast.message}</span>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
