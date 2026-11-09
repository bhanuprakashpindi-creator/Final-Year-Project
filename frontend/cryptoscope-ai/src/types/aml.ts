/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Anti-Money Laundering Types & Interfaces
 */

export type DatasetLabel = 'licit' | 'illicit' | 'unknown';
export type DatasetSplit = 'train' | 'val' | 'test';
export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';
export type TelemetryMode = 'live' | 'benchmark' | 'offline';

export type ModelId = 'random_forest' | 'gcn' | 'graphsage' | 'gatv2' | 'chebyshev';

export interface ModelPrediction {
  model_id: ModelId;
  model_name: string;
  model_type: 'Baseline Tabular' | 'Spatial GNN' | 'Inductive GNN' | 'Attention GNN' | 'Spectral GNN';
  illicit_prob: number;
  licit_prob: number;
  predicted_class: 'licit' | 'illicit';
  confidence: number;
}

export interface ConsensusResult {
  average_illicit_prob: number;
  average_licit_prob: number;
  consensus_class: 'licit' | 'illicit';
  agreement_ratio: string; // e.g. "5/5" or "4/5"
  risk_level: RiskLevel;
  illicit_votes: number;
  total_votes: number;
}

export interface EllipticTransaction {
  txId: string;
  timestep: number;
  split: DatasetSplit;
  ground_truth: DatasetLabel;
  in_degree: number;
  out_degree: number;
  features_sample?: {
    feature_name: string;
    value: number;
    description: string;
  }[];
  predictions?: Record<ModelId, ModelPrediction>;
  consensus?: ConsensusResult;
  provenance: 'live_model_inference' | 'verified_benchmark_record';
}

export interface GraphNode {
  id: string;
  label: DatasetLabel;
  timestep: number;
  is_central: boolean;
  hop: number;
  in_degree: number;
  out_degree: number;
  predicted_class?: 'licit' | 'illicit' | 'unknown';
  illicit_prob?: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  directed: boolean;
}

export interface TransactionGraphData {
  central_node_id: string;
  hops: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    total_nodes: number;
    total_edges: number;
    inflows: number;
    outflows: number;
    illicit_count: number;
    licit_count: number;
    unknown_count: number;
  };
  provenance: 'live_model_inference' | 'verified_benchmark_record';
}

export interface ModelBenchmarkMetric {
  model_id: ModelId;
  name: string;
  type: string;
  precision: number;
  recall: number;
  illicit_f1: number;
  macro_f1: number;
  pr_auc: number;
  accuracy: number;
  confusion_matrix: {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  };
}

export interface TrainingCurvePoint {
  epoch: number;
  rf_loss?: number;
  gcn_loss: number;
  gcn_f1: number;
  graphsage_loss: number;
  graphsage_f1: number;
  gatv2_loss: number;
  gatv2_f1: number;
  chebyshev_loss: number;
  chebyshev_f1: number;
}

export interface BackendHealth {
  status: 'online' | 'offline';
  apiUrl: string;
  latency_ms?: number;
  device?: string;
  models_loaded?: {
    random_forest: boolean;
    gcn: boolean;
    graphsage: boolean;
    gatv2: boolean;
    chebyshev: boolean;
  };
  graph_loaded?: boolean;
}
