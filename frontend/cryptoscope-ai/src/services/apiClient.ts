/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — API Client & Telemetry Service
 */

import {
  BackendHealth,
  EllipticTransaction,
  ModelBenchmarkMetric,
  TelemetryMode,
  TransactionGraphData,
} from '../types/aml';
import {
  BENCHMARK_SUBGRAPHS,
  BENCHMARK_TRANSACTIONS,
  DATASET_STATISTICS,
  MODEL_BENCHMARK_RESULTS,
} from '../data/ellipticBenchmark';

class ApiClient {
  private baseUrl: string = 'http://localhost:8000';
  private health: BackendHealth = {
    status: 'offline',
    apiUrl: 'http://localhost:8000',
  };
  private telemetryMode: TelemetryMode = 'benchmark';
  private listeners: ((health: BackendHealth, mode: TelemetryMode) => void)[] = [];
  private checkInterval: number | null = null;

  constructor() {
    const savedUrl = typeof window !== 'undefined' ? localStorage.getItem('cryptoscope_backend_url') : null;
    if (savedUrl) {
      this.baseUrl = savedUrl;
      this.health.apiUrl = savedUrl;
    }
  }

  public getBaseUrl(): string {
    return this.baseUrl;
  }

  public setBaseUrl(url: string): void {
    this.baseUrl = url.trim().replace(/\/+$/, '');
    this.health.apiUrl = this.baseUrl;
    if (typeof window !== 'undefined') {
      localStorage.setItem('cryptoscope_backend_url', this.baseUrl);
    }
    this.checkHealth();
  }

  public getHealth(): BackendHealth {
    return this.health;
  }

  public getTelemetryMode(): TelemetryMode {
    return this.telemetryMode;
  }

  public subscribe(listener: (health: BackendHealth, mode: TelemetryMode) => void): () => void {
    this.listeners.push(listener);
    listener(this.health, this.telemetryMode);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  private notify(): void {
    for (const listener of this.listeners) {
      listener(this.health, this.telemetryMode);
    }
  }

  public async checkHealth(): Promise<BackendHealth> {
    const startTime = performance.now();
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);

      const res = await fetch(`${this.baseUrl}/api/health`, {
        signal: controller.signal,
        headers: { Accept: 'application/json' },
      });
      clearTimeout(timeoutId);

      const latency_ms = Math.round(performance.now() - startTime);

      if (res.ok) {
        const data = await res.json();
        this.health = {
          status: 'online',
          apiUrl: this.baseUrl,
          latency_ms,
          device: data.device || 'cuda:0',
          models_loaded: data.models_loaded || {
            random_forest: true,
            gcn: true,
            graphsage: true,
            gatv2: true,
            chebyshev: true,
          },
          graph_loaded: data.graph_loaded ?? true,
        };
        this.telemetryMode = 'live';
      } else {
        this.health = {
          status: 'offline',
          apiUrl: this.baseUrl,
          latency_ms,
        };
        this.telemetryMode = 'benchmark';
      }
    } catch {
      this.health = {
        status: 'offline',
        apiUrl: this.baseUrl,
      };
      this.telemetryMode = 'benchmark';
    }

    this.notify();
    return this.health;
  }

  public startHealthPolling(intervalMs: number = 15000): void {
    this.checkHealth();
    if (this.checkInterval) clearInterval(this.checkInterval);
    this.checkInterval = window.setInterval(() => {
      this.checkHealth();
    }, intervalMs);
  }

  public stopHealthPolling(): void {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
  }

  /**
   * Fetch Transaction Details & 5-Model Predictions
   */
  public async getTransaction(txId: string): Promise<EllipticTransaction> {
    const cleanId = txId.trim();

    // 1. Try Live FastAPI backend if online
    if (this.health.status === 'online') {
      try {
        const res = await fetch(`${this.baseUrl}/api/transaction/${cleanId}`, {
          headers: { Accept: 'application/json' },
        });
        if (res.ok) {
          const data = await res.json();
          return {
            ...data,
            provenance: 'live_model_inference',
          };
        }
      } catch (e) {
        console.warn('Live backend query failed, falling back to verified benchmark suite:', e);
      }
    }

    // 2. Query Verified Benchmark Test Ledger
    if (BENCHMARK_TRANSACTIONS[cleanId]) {
      return BENCHMARK_TRANSACTIONS[cleanId];
    }

    // 3. Informative error if not in benchmark records and backend offline
    throw new Error(
      `Transaction ID "${cleanId}" was not found in the verified benchmark records, and the live Python FastAPI backend is currently offline at ${this.baseUrl}. Please select one of the verified benchmark test transactions or launch your Python backend to run live inference.`
    );
  }

  /**
   * Fetch 1-hop or 2-hop topological graph neighborhood
   */
  public async getNeighborhood(txId: string, hops: number = 1): Promise<TransactionGraphData> {
    const cleanId = txId.trim();

    // 1. Try Live FastAPI backend
    if (this.health.status === 'online') {
      try {
        const res = await fetch(`${this.baseUrl}/api/graph/${cleanId}?hops=${hops}`, {
          headers: { Accept: 'application/json' },
        });
        if (res.ok) {
          const data = await res.json();
          return {
            ...data,
            provenance: 'live_model_inference',
          };
        }
      } catch (e) {
        console.warn('Live graph query failed, checking benchmark subgraphs:', e);
      }
    }

    // 2. Query Verified Benchmark Subgraph
    if (BENCHMARK_SUBGRAPHS[cleanId]) {
      const full = BENCHMARK_SUBGRAPHS[cleanId];
      if (hops === 1) {
        // Filter down to 1-hop only
        const filteredNodes = full.nodes.filter((n) => n.hop <= 1);
        const nodeIds = new Set(filteredNodes.map((n) => n.id));
        const filteredEdges = full.edges.filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));
        return {
          ...full,
          hops: 1,
          nodes: filteredNodes,
          edges: filteredEdges,
          stats: {
            ...full.stats,
            total_nodes: filteredNodes.length,
            total_edges: filteredEdges.length,
          },
        };
      }
      return full;
    }

    // Check if node exists in benchmark transactions to synthesize its direct edges
    if (BENCHMARK_TRANSACTIONS[cleanId]) {
      // Return a 1-hop minimal verified topology
      const tx = BENCHMARK_TRANSACTIONS[cleanId];
      return {
        central_node_id: cleanId,
        hops: 1,
        provenance: 'verified_benchmark_record',
        nodes: [
          {
            id: cleanId,
            label: tx.ground_truth,
            timestep: tx.timestep,
            is_central: true,
            hop: 0,
            in_degree: tx.in_degree,
            out_degree: tx.out_degree,
            predicted_class: tx.consensus?.consensus_class,
            illicit_prob: tx.consensus?.average_illicit_prob,
          },
        ],
        edges: [],
        stats: {
          total_nodes: 1,
          total_edges: 0,
          inflows: tx.in_degree,
          outflows: tx.out_degree,
          illicit_count: tx.ground_truth === 'illicit' ? 1 : 0,
          licit_count: tx.ground_truth === 'licit' ? 1 : 0,
          unknown_count: tx.ground_truth === 'unknown' ? 1 : 0,
        },
      };
    }

    throw new Error(
      `Transaction graph for ID "${cleanId}" cannot be rendered because the live Python backend is offline and this ID is not in the bundled benchmark subgraph index.`
    );
  }

  /**
   * Fetch 5-Model Benchmark Metrics
   */
  public async getMetrics(): Promise<ModelBenchmarkMetric[]> {
    if (this.health.status === 'online') {
      try {
        const res = await fetch(`${this.baseUrl}/api/metrics`);
        if (res.ok) {
          return await res.json();
        }
      } catch (e) {
        console.warn('Failed to fetch live metrics, using authentic benchmark results:', e);
      }
    }
    return MODEL_BENCHMARK_RESULTS;
  }

  /**
   * Fetch Dataset Statistics
   */
  public async getDatasetStats(): Promise<typeof DATASET_STATISTICS> {
    if (this.health.status === 'online') {
      try {
        const res = await fetch(`${this.baseUrl}/api/dataset/stats`);
        if (res.ok) {
          return await res.json();
        }
      } catch (e) {
        console.warn('Failed to fetch live stats, using authentic dataset statistics:', e);
      }
    }
    return DATASET_STATISTICS;
  }
}

export const apiClient = new ApiClient();
