/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Investigate Page (Google Stitch Design)
 */

import React, { useEffect, useState } from 'react';
import {
  Search,
  AlertTriangle,
  CheckCircle2,
  Share2,
  Layers,
  ArrowRight,
  ShieldAlert,
  Clock,
  Database,
  Radio,
  Copy,
  Check,
  RefreshCw,
} from 'lucide-react';
import { EllipticTransaction } from '../types/aml';
import { apiClient } from '../services/apiClient';
import { ModelPredictionCard } from '../components/ModelPredictionCard';
import { ProbabilityBar } from '../components/ProbabilityBar';

interface InvestigatePageProps {
  currentTxId: string;
  onExploreGraph: (txId: string) => void;
}

export const InvestigatePage: React.FC<InvestigatePageProps> = ({
  currentTxId,
  onExploreGraph,
}) => {
  const [txIdInput, setTxIdInput] = useState(currentTxId || '230425980');
  const [transaction, setTransaction] = useState<EllipticTransaction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const fetchInvestigation = async (id: string) => {
    if (!id.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.getTransaction(id.trim());
      setTransaction(data);
    } catch (err: any) {
      setTransaction(null);
      setError(err.message || 'Failed to retrieve transaction analysis.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (currentTxId) {
      setTxIdInput(currentTxId);
      fetchInvestigation(currentTxId);
    } else {
      fetchInvestigation('230425980');
    }
  }, [currentTxId]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchInvestigation(txIdInput);
  };

  const handleCopyId = () => {
    if (transaction) {
      navigator.clipboard.writeText(transaction.txId);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const getRiskBadge = (risk?: string) => {
    switch (risk) {
      case 'critical':
        return { label: 'CRITICAL RISK', color: 'bg-[#EF4444]/15 text-[#EF4444] border-[#EF4444]/30' };
      case 'high':
        return { label: 'HIGH RISK', color: 'bg-[#F97316]/15 text-[#F97316] border-[#F97316]/30' };
      case 'medium':
        return { label: 'MEDIUM RISK', color: 'bg-[#F59E0B]/15 text-[#F59E0B] border-[#F59E0B]/30' };
      case 'low':
      default:
        return { label: 'LOW RISK', color: 'bg-[#10B981]/15 text-[#10B981] border-[#10B981]/30' };
    }
  };

  const riskBadge = getRiskBadge(transaction?.consensus?.risk_level);

  return (
    <div className="space-y-8 py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto animate-fade-in">
      {/* Top Search & Presets Bar */}
      <section className="bg-[#111827] border border-[#1E293B] rounded-xl p-5 space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-[#F8FAFC]">Forensic Transaction Investigator</h2>
            <p className="text-xs text-[#94A3B8]">
              Input an Elliptic Dataset Transaction ID to run five-model inference and consensus triage.
            </p>
          </div>

          {/* Quick-Pick Test Case Presets */}
          <div className="flex items-center gap-1.5 flex-wrap text-xs font-mono">
            <span className="text-[#64748B] text-[11px] uppercase mr-1">Presets:</span>
            <button
              onClick={() => {
                setTxIdInput('230425980');
                fetchInvestigation('230425980');
              }}
              className="px-2 py-1 rounded bg-[#EF4444]/10 hover:bg-[#EF4444]/20 text-[#EF4444] border border-[#EF4444]/30 transition-colors"
            >
              Illicit #230425980
            </button>
            <button
              onClick={() => {
                setTxIdInput('5530414');
                fetchInvestigation('5530414');
              }}
              className="px-2 py-1 rounded bg-[#10B981]/10 hover:bg-[#10B981]/20 text-[#10B981] border border-[#10B981]/30 transition-colors"
            >
              Licit #5530414
            </button>
            <button
              onClick={() => {
                setTxIdInput('232022460');
                fetchInvestigation('232022460');
              }}
              className="px-2 py-1 rounded bg-[#F59E0B]/10 hover:bg-[#F59E0B]/20 text-[#F59E0B] border border-[#F59E0B]/30 transition-colors"
            >
              Mixer #232022460
            </button>
          </div>
        </div>

        {/* Input Form */}
        <form onSubmit={handleSearchSubmit} className="flex gap-2">
          <div className="relative flex-1">
            <input
              type="text"
              value={txIdInput}
              onChange={(e) => setTxIdInput(e.target.value)}
              placeholder="Enter Elliptic Transaction ID (e.g. 230425980)..."
              className="w-full h-11 pl-10 pr-4 bg-[#080C14] border border-[#1E293B] rounded-lg text-xs font-mono text-[#F8FAFC] placeholder-[#475569] focus:outline-none focus:border-[#00F2FE] focus:ring-1 focus:ring-[#00F2FE]"
            />
            <Search className="w-4 h-4 text-[#64748B] absolute left-3.5 top-3.5 pointer-events-none" />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="px-5 h-11 bg-[#00F2FE] hover:bg-[#38BDF8] text-[#080C14] text-xs font-semibold rounded-lg flex items-center gap-1.5 transition-colors shrink-0 disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            <span>Analyze</span>
          </button>
        </form>
      </section>

      {/* Loading Skeleton */}
      {loading && (
        <div className="p-12 text-center bg-[#111827] border border-[#1E293B] rounded-xl space-y-3">
          <RefreshCw className="w-8 h-8 text-[#00F2FE] animate-spin mx-auto" />
          <p className="text-sm font-semibold text-[#F8FAFC]">Running Five-Model AML Inference Pipeline...</p>
          <p className="text-xs text-[#94A3B8]">Passing transaction features through Random Forest and GNN checkpoints</p>
        </div>
      )}

      {/* Error Notice */}
      {error && !loading && (
        <div className="p-6 bg-[#EF4444]/10 border border-[#EF4444]/30 rounded-xl space-y-2 text-xs">
          <div className="flex items-center gap-2 text-[#EF4444] font-semibold text-sm">
            <AlertTriangle className="w-5 h-5" />
            <span>Investigation Error</span>
          </div>
          <p className="text-[#F8FAFC] leading-relaxed">{error}</p>
          <p className="text-[#94A3B8] pt-1">
            Tip: When offline, select one of the verified benchmark test transactions (e.g. <strong>#230425980</strong>, <strong>#5530414</strong>, or <strong>#232022460</strong>) to explore authentic results.
          </p>
        </div>
      )}

      {/* Investigation Results */}
      {transaction && !loading && (
        <div className="space-y-6">
          {/* Header Card: Transaction Identity & Consensus HUD */}
          <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-6">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 pb-6 border-b border-[#1E293B]">
              {/* Left Identity */}
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[11px] font-mono text-[#64748B] uppercase">Elliptic Dataset TxID:</span>
                  <div className="flex items-center gap-1.5 font-mono text-lg font-bold text-[#F8FAFC]">
                    <span>{transaction.txId}</span>
                    <button
                      onClick={handleCopyId}
                      className="p-1 text-[#94A3B8] hover:text-[#00F2FE] rounded transition-colors"
                      title="Copy TxID"
                    >
                      {copied ? <Check className="w-4 h-4 text-[#10B981]" /> : <Copy className="w-4 h-4" />}
                    </button>
                  </div>

                  {/* Provenance Badge */}
                  <span
                    className={`ml-2 text-[10px] font-mono px-2 py-0.5 rounded border uppercase ${
                      transaction.provenance === 'live_model_inference'
                        ? 'bg-[#10B981]/15 text-[#10B981] border-[#10B981]/40'
                        : 'bg-[#3B82F6]/15 text-[#ADC6FF] border-[#3B82F6]/40'
                    }`}
                  >
                    {transaction.provenance === 'live_model_inference' ? '🟢 LIVE INFERENCE' : '🔵 VERIFIED BENCHMARK'}
                  </span>
                </div>

                <div className="flex flex-wrap items-center gap-4 text-xs font-mono text-[#94A3B8]">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5 text-[#00F2FE]" />
                    <span>Timestep: <strong className="text-[#F8FAFC]">t = {transaction.timestep}</strong></span>
                  </span>
                  <span>·</span>
                  <span className="flex items-center gap-1">
                    <Database className="w-3.5 h-3.5 text-[#00F2FE]" />
                    <span>Dataset Split: <strong className="text-[#F8FAFC] uppercase">{transaction.split}</strong></span>
                  </span>
                  <span>·</span>
                  <span>
                    Ground Truth: <strong className="text-[#F8FAFC] uppercase">{transaction.ground_truth}</strong>
                  </span>
                </div>
              </div>

              {/* Right: Pivot to Network Explorer Button */}
              <div>
                <button
                  onClick={() => onExploreGraph(transaction.txId)}
                  className="px-4 py-2.5 bg-[#1E293B] hover:bg-[#334155] hover:border-[#00F2FE]/40 border border-[#334155] text-xs font-mono font-medium text-[#F8FAFC] rounded-lg flex items-center gap-2 transition-all group shadow-sm"
                >
                  <Share2 className="w-4 h-4 text-[#00F2FE] group-hover:scale-110 transition-transform" />
                  <span>Explore in Network Graph</span>
                  <ArrowRight className="w-3.5 h-3.5 text-[#64748B]" />
                </button>
              </div>
            </div>

            {/* Consensus Triage Strip */}
            {transaction.consensus && (
              <div className="pt-6 grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
                {/* Risk Level */}
                <div className="p-3.5 bg-[#080C14] border border-[#1E293B] rounded-lg">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#64748B] block mb-1">
                    Risk Classification
                  </span>
                  <span className={`inline-block px-2.5 py-1 rounded text-xs font-mono font-bold border ${riskBadge.color}`}>
                    {riskBadge.label}
                  </span>
                </div>

                {/* Average Illicit Probability */}
                <div className="p-3.5 bg-[#080C14] border border-[#1E293B] rounded-lg">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#64748B] block mb-1">
                    Average Illicit Probability
                  </span>
                  <span className="text-xl font-bold font-mono text-[#EF4444] tabular-nums">
                    {(transaction.consensus.average_illicit_prob * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Model Agreement */}
                <div className="p-3.5 bg-[#080C14] border border-[#1E293B] rounded-lg">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#64748B] block mb-1">
                    Five-Model Agreement
                  </span>
                  <span className="text-xl font-bold font-mono text-[#F8FAFC]">
                    {transaction.consensus.agreement_ratio}{' '}
                    <span className="text-xs font-normal text-[#94A3B8]">
                      ({transaction.consensus.illicit_votes} Illicit / {transaction.consensus.total_votes} Total)
                    </span>
                  </span>
                </div>

                {/* Consensus Class */}
                <div className="p-3.5 bg-[#080C14] border border-[#1E293B] rounded-lg">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#64748B] block mb-1">
                    Consensus Verdict
                  </span>
                  <span
                    className={`text-base font-bold font-mono uppercase ${
                      transaction.consensus.consensus_class === 'illicit' ? 'text-[#EF4444]' : 'text-[#10B981]'
                    }`}
                  >
                    {transaction.consensus.consensus_class}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Five Model Predictions Grid */}
          <div className="space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-[#1E293B]">
              <div>
                <h3 className="text-base font-semibold text-[#F8FAFC]">Five-Model Prediction Telemetry</h3>
                <p className="text-xs text-[#94A3B8]">
                  Individual model scores across Random Forest baseline and four Graph Neural Networks.
                </p>
              </div>
            </div>

            {transaction.predictions ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                <ModelPredictionCard prediction={transaction.predictions.random_forest} />
                <ModelPredictionCard prediction={transaction.predictions.gcn} />
                <ModelPredictionCard prediction={transaction.predictions.graphsage} />
                <ModelPredictionCard prediction={transaction.predictions.gatv2} />
                <ModelPredictionCard prediction={transaction.predictions.chebyshev} />
              </div>
            ) : (
              <div className="p-6 bg-[#111827] border border-[#1E293B] rounded-xl text-center text-xs text-[#94A3B8]">
                Prediction breakdown unavailable for this record.
              </div>
            )}
          </div>

          {/* Sample Features Audit Table */}
          {transaction.features_sample && transaction.features_sample.length > 0 && (
            <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-6 space-y-4">
              <div>
                <h3 className="text-base font-semibold text-[#F8FAFC]">Transaction Feature Vectors</h3>
                <p className="text-xs text-[#94A3B8]">
                  Representative local and 1-hop aggregated graph features extracted from the 145 reduced Elliptic dimensions.
                </p>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-[#0B0F19] text-[#64748B] font-mono uppercase text-[10px] border-b border-[#1E293B]">
                    <tr>
                      <th className="py-2.5 px-3">Feature Name</th>
                      <th className="py-2.5 px-3 text-right">Normalized Value</th>
                      <th className="py-2.5 px-3">Semantic Description</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1E293B] font-mono">
                    {transaction.features_sample.map((feat, idx) => (
                      <tr key={idx} className="hover:bg-[#1E293B]/40 transition-colors">
                        <td className="py-2.5 px-3 text-[#00F2FE] font-medium">{feat.feature_name}</td>
                        <td className="py-2.5 px-3 text-right text-[#F8FAFC] tabular-nums font-bold">
                          {typeof feat.value === 'number' ? feat.value.toFixed(4) : feat.value}
                        </td>
                        <td className="py-2.5 px-3 text-[#94A3B8] font-sans">{feat.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
