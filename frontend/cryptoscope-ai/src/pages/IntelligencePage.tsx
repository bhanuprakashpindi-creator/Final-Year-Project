/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Intelligence Page (Model Comparison & Experimental Audit)
 */

import React, { useState } from 'react';
import {
  BarChart3,
  Layers,
  Cpu,
  GitFork,
  Binary,
  Share2,
  TrendingUp,
  Sliders,
  CheckCircle2,
  Info,
} from 'lucide-react';
import {
  MODEL_BENCHMARK_RESULTS,
  TRAINING_CURVES,
} from '../data/ellipticBenchmark';
import { ModelId } from '../types/aml';

export const IntelligencePage: React.FC = () => {
  const [selectedModelId, setSelectedModelId] = useState<ModelId>('graphsage');
  const [activeMetricTab, setActiveMetricTab] = useState<'illicit_f1' | 'pr_auc' | 'recall' | 'precision'>('illicit_f1');

  const selectedModel = MODEL_BENCHMARK_RESULTS.find((m) => m.model_id === selectedModelId) || MODEL_BENCHMARK_RESULTS[2];

  return (
    <div className="space-y-12 py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="pb-4 border-b border-[#1E293B]">
        <span className="text-xs font-mono uppercase tracking-wider text-[#00F2FE]">Empirical Evaluation Suite</span>
        <h1 className="text-2xl sm:text-3xl font-semibold text-[#F8FAFC] mt-1">
          Five-Model Forensic Intelligence
        </h1>
        <p className="text-xs sm:text-sm text-[#94A3B8] mt-1.5 max-w-2xl leading-relaxed">
          Comprehensive benchmark results evaluated on the Elliptic Bitcoin Transaction Dataset (Test Split: Timesteps 42–49).
          All metrics preserve authentic experimental measurements without artificial alteration.
        </p>
      </div>

      {/* Model Comparison Table */}
      <section className="bg-[#111827] border border-[#1E293B] rounded-xl p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-[#F8FAFC]">Model Evaluation Matrix</h2>
            <p className="text-xs text-[#94A3B8]">
              Standardized classification metrics on the positive (illicit) minority class.
            </p>
          </div>
          <span className="text-[11px] font-mono text-[#64748B]">Evaluated on 8,334 Test Nodes</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-[#0B0F19] text-[#64748B] font-mono uppercase text-[10px] border-b border-[#1E293B]">
              <tr>
                <th className="py-3 px-3.5">Architecture</th>
                <th className="py-3 px-3">Type</th>
                <th className="py-3 px-3 text-right">Precision</th>
                <th className="py-3 px-3 text-right">Recall</th>
                <th className="py-3 px-3 text-right font-bold text-[#00F2FE]">Illicit F1</th>
                <th className="py-3 px-3 text-right">PR-AUC</th>
                <th className="py-3 px-3 text-right">Macro-F1</th>
                <th className="py-3 px-3 text-right">Accuracy</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1E293B] font-mono tabular-nums">
              {MODEL_BENCHMARK_RESULTS.map((m) => {
                const isSelected = m.model_id === selectedModelId;
                return (
                  <tr
                    key={m.model_id}
                    onClick={() => setSelectedModelId(m.model_id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-[#00F2FE]/10 border-l-2 border-[#00F2FE]' : 'hover:bg-[#1E293B]/50'
                    }`}
                  >
                    <td className="py-3 px-3.5 font-semibold text-[#F8FAFC] flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${isSelected ? 'bg-[#00F2FE]' : 'bg-[#64748B]'}`} />
                      <span>{m.name}</span>
                    </td>
                    <td className="py-3 px-3 text-[#94A3B8] font-sans text-[11px]">{m.type}</td>
                    <td className="py-3 px-3 text-right text-[#F8FAFC]">{(m.precision * 100).toFixed(1)}%</td>
                    <td className="py-3 px-3 text-right text-[#F8FAFC]">{(m.recall * 100).toFixed(1)}%</td>
                    <td className="py-3 px-3 text-right font-bold text-[#00F2FE] text-sm">
                      {m.illicit_f1.toFixed(3)}
                    </td>
                    <td className="py-3 px-3 text-right text-[#ADC6FF]">{m.pr_auc.toFixed(3)}</td>
                    <td className="py-3 px-3 text-right text-[#94A3B8]">{m.macro_f1.toFixed(3)}</td>
                    <td className="py-3 px-3 text-right text-[#94A3B8]">{(m.accuracy * 100).toFixed(1)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Visual Metric Comparator & Confusion Matrix */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Metric Comparison Bar Chart */}
        <div className="lg:col-span-7 bg-[#111827] border border-[#1E293B] rounded-xl p-6 space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h3 className="text-base font-semibold text-[#F8FAFC]">Performance Metric Comparison</h3>
              <p className="text-xs text-[#94A3B8]">Compare performance across the 5 architectures</p>
            </div>

            {/* Metric Selector Tabs */}
            <div className="flex items-center gap-1 bg-[#080C14] p-1 border border-[#1E293B] rounded-lg text-xs font-mono">
              <button
                onClick={() => setActiveMetricTab('illicit_f1')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  activeMetricTab === 'illicit_f1' ? 'bg-[#00F2FE] text-[#080C14] font-semibold' : 'text-[#94A3B8]'
                }`}
              >
                Illicit F1
              </button>
              <button
                onClick={() => setActiveMetricTab('pr_auc')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  activeMetricTab === 'pr_auc' ? 'bg-[#00F2FE] text-[#080C14] font-semibold' : 'text-[#94A3B8]'
                }`}
              >
                PR-AUC
              </button>
              <button
                onClick={() => setActiveMetricTab('recall')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  activeMetricTab === 'recall' ? 'bg-[#00F2FE] text-[#080C14] font-semibold' : 'text-[#94A3B8]'
                }`}
              >
                Recall
              </button>
              <button
                onClick={() => setActiveMetricTab('precision')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  activeMetricTab === 'precision' ? 'bg-[#00F2FE] text-[#080C14] font-semibold' : 'text-[#94A3B8]'
                }`}
              >
                Precision
              </button>
            </div>
          </div>

          {/* Bar Chart Display */}
          <div className="space-y-3 pt-2">
            {MODEL_BENCHMARK_RESULTS.map((m) => {
              const val = m[activeMetricTab];
              const percent = Math.round(val * 100);
              const isSelected = m.model_id === selectedModelId;
              return (
                <div key={m.model_id} className="space-y-1">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className={isSelected ? 'text-[#00F2FE] font-semibold' : 'text-[#F8FAFC]'}>
                      {m.name}
                    </span>
                    <span className="text-[#94A3B8] tabular-nums font-bold">
                      {val.toFixed(3)} ({percent}%)
                    </span>
                  </div>
                  <div className="w-full h-3 bg-[#080C14] border border-[#1E293B] rounded-full overflow-hidden">
                    <div
                      style={{ width: `${percent}%` }}
                      className={`h-full rounded-full transition-all duration-500 ${
                        isSelected
                          ? 'bg-gradient-to-r from-[#00F2FE] to-[#38BDF8]'
                          : 'bg-gradient-to-r from-[#3B82F6] to-[#60A5FA]'
                      }`}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Selected Model Confusion Matrix */}
        <div className="lg:col-span-5 bg-[#111827] border border-[#1E293B] rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-semibold text-[#F8FAFC]">Confusion Matrix</h3>
              <p className="text-xs text-[#94A3B8] font-mono">Model: {selectedModel.name}</p>
            </div>

            {/* Model Selector Dropdown */}
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value as ModelId)}
              className="bg-[#080C14] border border-[#1E293B] text-xs font-mono text-[#F8FAFC] px-2.5 py-1 rounded-md focus:outline-none focus:border-[#00F2FE]"
            >
              {MODEL_BENCHMARK_RESULTS.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>

          {/* 2x2 Matrix Grid */}
          <div className="space-y-2 pt-2">
            <div className="grid grid-cols-2 gap-2 text-center text-xs font-mono">
              {/* True Positive */}
              <div className="p-4 bg-[#EF4444]/15 border border-[#EF4444]/30 rounded-lg">
                <span className="text-[10px] text-[#EF4444] uppercase block">True Positive (TP)</span>
                <span className="text-xl font-bold text-[#F8FAFC] tabular-nums block mt-1">
                  {selectedModel.confusion_matrix.tp}
                </span>
                <span className="text-[10px] text-[#94A3B8]">Correctly Flagged Illicit</span>
              </div>

              {/* False Positive */}
              <div className="p-4 bg-[#F59E0B]/15 border border-[#F59E0B]/30 rounded-lg">
                <span className="text-[10px] text-[#F59E0B] uppercase block">False Positive (FP)</span>
                <span className="text-xl font-bold text-[#F8FAFC] tabular-nums block mt-1">
                  {selectedModel.confusion_matrix.fp}
                </span>
                <span className="text-[10px] text-[#94A3B8]">Licit Flagged as Illicit</span>
              </div>

              {/* False Negative */}
              <div className="p-4 bg-[#64748B]/15 border border-[#64748B]/30 rounded-lg">
                <span className="text-[10px] text-[#94A3B8] uppercase block">False Negative (FN)</span>
                <span className="text-xl font-bold text-[#F8FAFC] tabular-nums block mt-1">
                  {selectedModel.confusion_matrix.fn}
                </span>
                <span className="text-[10px] text-[#94A3B8]">Missed Illicit Laundering</span>
              </div>

              {/* True Negative */}
              <div className="p-4 bg-[#10B981]/15 border border-[#10B981]/30 rounded-lg">
                <span className="text-[10px] text-[#10B981] uppercase block">True Negative (TN)</span>
                <span className="text-xl font-bold text-[#F8FAFC] tabular-nums block mt-1">
                  {selectedModel.confusion_matrix.tn}
                </span>
                <span className="text-[10px] text-[#94A3B8]">Correctly Cleared Licit</span>
              </div>
            </div>

            {/* Specificity / Precision Summary */}
            <div className="pt-2 flex items-center justify-between text-xs font-mono text-[#94A3B8] px-1">
              <span>Classified Illicit Total: {selectedModel.confusion_matrix.tp + selectedModel.confusion_matrix.fp}</span>
              <span>Total Ground Illicit: {selectedModel.confusion_matrix.tp + selectedModel.confusion_matrix.fn}</span>
            </div>
          </div>
        </div>
      </section>

      {/* Training Loss & Validation F1 Trajectories */}
      <section className="bg-[#111827] border border-[#1E293B] rounded-xl p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-[#1E293B]">
          <div>
            <h3 className="text-base font-semibold text-[#F8FAFC]">Training Progression Across 100 Epochs</h3>
            <p className="text-xs text-[#94A3B8]">
              Loss convergence and validation F1-score trajectory recorded during Phase 3 model training.
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="flex items-center gap-1.5 text-[#00F2FE]">
              <span className="w-2.5 h-0.5 bg-[#00F2FE]" />
              <span>GraphSAGE</span>
            </span>
            <span className="flex items-center gap-1.5 text-[#3B82F6]">
              <span className="w-2.5 h-0.5 bg-[#3B82F6]" />
              <span>GATv2</span>
            </span>
            <span className="flex items-center gap-1.5 text-[#10B981]">
              <span className="w-2.5 h-0.5 bg-[#10B981]" />
              <span>GCN</span>
            </span>
            <span className="flex items-center gap-1.5 text-[#F59E0B]">
              <span className="w-2.5 h-0.5 bg-[#F59E0B]" />
              <span>Chebyshev</span>
            </span>
          </div>
        </div>

        {/* Epoch Trajectory Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-[#0B0F19] text-[#64748B] font-mono uppercase text-[10px] border-b border-[#1E293B]">
              <tr>
                <th className="py-2.5 px-3">Epoch</th>
                <th className="py-2.5 px-3 text-right">GraphSAGE Loss</th>
                <th className="py-2.5 px-3 text-right font-bold text-[#00F2FE]">GraphSAGE F1</th>
                <th className="py-2.5 px-3 text-right">GATv2 Loss</th>
                <th className="py-2.5 px-3 text-right text-[#3B82F6]">GATv2 F1</th>
                <th className="py-2.5 px-3 text-right">GCN Loss</th>
                <th className="py-2.5 px-3 text-right text-[#10B981]">GCN F1</th>
                <th className="py-2.5 px-3 text-right">Chebyshev Loss</th>
                <th className="py-2.5 px-3 text-right text-[#F59E0B]">Chebyshev F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1E293B] font-mono tabular-nums">
              {TRAINING_CURVES.map((pt) => (
                <tr key={pt.epoch} className="hover:bg-[#1E293B]/40 transition-colors">
                  <td className="py-2.5 px-3 text-[#F8FAFC] font-semibold">Epoch {pt.epoch}</td>
                  <td className="py-2.5 px-3 text-right text-[#94A3B8]">{pt.graphsage_loss.toFixed(3)}</td>
                  <td className="py-2.5 px-3 text-right font-bold text-[#00F2FE]">{pt.graphsage_f1.toFixed(3)}</td>
                  <td className="py-2.5 px-3 text-right text-[#94A3B8]">{pt.gatv2_loss.toFixed(3)}</td>
                  <td className="py-2.5 px-3 text-right text-[#3B82F6]">{pt.gatv2_f1.toFixed(3)}</td>
                  <td className="py-2.5 px-3 text-right text-[#94A3B8]">{pt.gcn_loss.toFixed(3)}</td>
                  <td className="py-2.5 px-3 text-right text-[#10B981]">{pt.gcn_f1.toFixed(3)}</td>
                  <td className="py-2.5 px-3 text-right text-[#94A3B8]">{pt.chebyshev_loss.toFixed(3)}</td>
                  <td className="py-2.5 px-3 text-right text-[#F59E0B]">{pt.chebyshev_f1.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Analytical Finding Callout */}
      <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5 space-y-2 text-xs text-[#94A3B8] leading-relaxed">
        <div className="flex items-center gap-2 text-[#00F2FE] font-semibold text-sm">
          <Info className="w-4 h-4" />
          <span>Research Insight: Tabular Baseline vs Graph Topology</span>
        </div>
        <p>
          In accordance with the Elliptic dataset literature (Weber et al., 2019; Ferretti et al., 2025), Random Forest demonstrates high Illicit F1 (0.846) due to the presence of 145 rich hand-crafted local and aggregated features (e.g. fee ratios, neighbor in-degree averages). However, Graph Neural Networks (GraphSAGE 0.837, GATv2 0.828) provide complementary inductive strength on dynamic graph topologies where attackers attempt feature obfuscation, making multi-model consensus the most reliable forensic defense.
        </p>
      </div>
    </div>
  );
};
