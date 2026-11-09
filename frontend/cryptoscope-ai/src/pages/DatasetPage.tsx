/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Dataset Page (Elliptic Bitcoin Transaction Dataset Audit)
 */

import React from 'react';
import {
  Database,
  Share2,
  Clock,
  Layers,
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  ArrowRight,
  TrendingDown,
  PieChart,
} from 'lucide-react';
import { DATASET_STATISTICS } from '../data/ellipticBenchmark';

export const DatasetPage: React.FC = () => {
  const stats = DATASET_STATISTICS;

  return (
    <div className="space-y-12 py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="pb-4 border-b border-[#1E293B]">
        <span className="text-xs font-mono uppercase tracking-wider text-[#00F2FE]">Dataset Architecture</span>
        <h1 className="text-2xl sm:text-3xl font-semibold text-[#F8FAFC] mt-1">
          Elliptic Bitcoin Transaction Dataset
        </h1>
        <p className="text-xs sm:text-sm text-[#94A3B8] mt-1.5 max-w-2xl leading-relaxed">
          The primary benchmark graph dataset for cryptocurrency anti-money laundering research, capturing real Bitcoin transaction flows across 49 discrete temporal intervals.
        </p>
      </div>

      {/* Macro Statistics KPI Grid */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5">
          <div className="flex items-center gap-2 text-[#64748B] text-xs font-mono mb-2">
            <Database className="w-4 h-4 text-[#00F2FE]" />
            <span>Total Transactions</span>
          </div>
          <span className="text-2xl font-bold font-mono text-[#F8FAFC] tabular-nums">
            {stats.total_transactions.toLocaleString()}
          </span>
          <span className="text-[11px] text-[#94A3B8] block mt-1">Graph Vertices ($V$)</span>
        </div>

        <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5">
          <div className="flex items-center gap-2 text-[#64748B] text-xs font-mono mb-2">
            <Share2 className="w-4 h-4 text-[#00F2FE]" />
            <span>Directed Edges</span>
          </div>
          <span className="text-2xl font-bold font-mono text-[#F8FAFC] tabular-nums">
            {stats.total_edges.toLocaleString()}
          </span>
          <span className="text-[11px] text-[#94A3B8] block mt-1">UTXO Payment Flows ($E$)</span>
        </div>

        <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5">
          <div className="flex items-center gap-2 text-[#64748B] text-xs font-mono mb-2">
            <Clock className="w-4 h-4 text-[#00F2FE]" />
            <span>Discrete Timesteps</span>
          </div>
          <span className="text-2xl font-bold font-mono text-[#F8FAFC] tabular-nums">
            {stats.total_timesteps}
          </span>
          <span className="text-[11px] text-[#94A3B8] block mt-1">~2-Week Time Intervals</span>
        </div>

        <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5">
          <div className="flex items-center gap-2 text-[#64748B] text-xs font-mono mb-2">
            <Layers className="w-4 h-4 text-[#00F2FE]" />
            <span>Reduced Features</span>
          </div>
          <span className="text-2xl font-bold font-mono text-[#F8FAFC] tabular-nums">
            {stats.reduced_features}
          </span>
          <span className="text-[11px] text-[#94A3B8] block mt-1">Normalized Dimensions</span>
        </div>
      </section>

      {/* Class Distribution Breakdown */}
      <section className="bg-[#111827] border border-[#1E293B] rounded-xl p-6 space-y-6">
        <div>
          <h2 className="text-base font-semibold text-[#F8FAFC]">Transaction Class Distribution</h2>
          <p className="text-xs text-[#94A3B8]">
            Extreme class imbalance characteristic of real-world financial fraud surveillance.
          </p>
        </div>

        {/* Visual Distribution Stacked Bar */}
        <div className="space-y-2">
          <div className="w-full h-4 bg-[#080C14] border border-[#1E293B] rounded-full overflow-hidden flex">
            <div
              style={{ width: `${stats.classes.illicit.percentage}%` }}
              className="h-full bg-[#EF4444] transition-all"
              title={`Illicit: ${stats.classes.illicit.percentage}%`}
            />
            <div
              style={{ width: `${stats.classes.licit.percentage}%` }}
              className="h-full bg-[#10B981] transition-all"
              title={`Licit: ${stats.classes.licit.percentage}%`}
            />
            <div
              style={{ width: `${stats.classes.unknown.percentage}%` }}
              className="h-full bg-[#334155] transition-all"
              title={`Unknown: ${stats.classes.unknown.percentage}%`}
            />
          </div>

          <div className="flex items-center justify-between text-xs font-mono text-[#94A3B8]">
            <span>Total Nodes: 203,769</span>
            <span>Labeled Subset: 46,564 (22.85%)</span>
          </div>
        </div>

        {/* Class Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-[#080C14] border border-[#1E293B] rounded-lg p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-semibold text-[#EF4444] flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-[#EF4444]" />
                <span>Class 1: Illicit</span>
              </span>
              <span className="text-xs font-mono text-[#EF4444] font-bold">{stats.classes.illicit.percentage}%</span>
            </div>
            <div className="text-xl font-bold font-mono text-[#F8FAFC] tabular-nums">
              {stats.classes.illicit.count.toLocaleString()}
            </div>
            <p className="text-xs text-[#94A3B8] leading-relaxed">{stats.classes.illicit.description}</p>
          </div>

          <div className="bg-[#080C14] border border-[#1E293B] rounded-lg p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-semibold text-[#10B981] flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-[#10B981]" />
                <span>Class 2: Licit</span>
              </span>
              <span className="text-xs font-mono text-[#10B981] font-bold">{stats.classes.licit.percentage}%</span>
            </div>
            <div className="text-xl font-bold font-mono text-[#F8FAFC] tabular-nums">
              {stats.classes.licit.count.toLocaleString()}
            </div>
            <p className="text-xs text-[#94A3B8] leading-relaxed">{stats.classes.licit.description}</p>
          </div>

          <div className="bg-[#080C14] border border-[#1E293B] rounded-lg p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-semibold text-[#94A3B8] flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-[#64748B]" />
                <span>Unknown / Unlabeled</span>
              </span>
              <span className="text-xs font-mono text-[#94A3B8] font-bold">{stats.classes.unknown.percentage}%</span>
            </div>
            <div className="text-xl font-bold font-mono text-[#F8FAFC] tabular-nums">
              {stats.classes.unknown.count.toLocaleString()}
            </div>
            <p className="text-xs text-[#94A3B8] leading-relaxed">{stats.classes.unknown.description}</p>
          </div>
        </div>
      </section>

      {/* Temporal Splitting Architecture */}
      <section className="bg-[#111827] border border-[#1E293B] rounded-xl p-6 space-y-6">
        <div>
          <h2 className="text-base font-semibold text-[#F8FAFC]">Temporal Splitting Protocol</h2>
          <p className="text-xs text-[#94A3B8]">
            Strict forward temporal partition preserving chronological arrow of time.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-[#080C14] border border-[#1E293B] rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase text-[#00F2FE] font-semibold">Training Split</span>
              <span className="text-xs font-mono text-[#94A3B8]">Timesteps {stats.splits.train.timesteps}</span>
            </div>
            <div className="space-y-1 font-mono text-xs">
              <div className="flex justify-between text-[#94A3B8]">
                <span>Total Nodes:</span>
                <span className="text-[#F8FAFC]">{stats.splits.train.total_nodes.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-[#94A3B8]">
                <span>Labeled Subset:</span>
                <span className="text-[#F8FAFC]">{stats.splits.train.labeled_nodes.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-[#94A3B8]">
                <span>Illicit Nodes:</span>
                <span className="text-[#EF4444]">{stats.splits.train.illicit_nodes.toLocaleString()}</span>
              </div>
            </div>
            <span className="text-[11px] font-mono text-[#64748B] block pt-1 border-t border-[#1E293B]">
              {stats.splits.train.ratio}
            </span>
          </div>

          <div className="bg-[#080C14] border border-[#1E293B] rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase text-[#3B82F6] font-semibold">Validation Split</span>
              <span className="text-xs font-mono text-[#94A3B8]">Timesteps {stats.splits.validation.timesteps}</span>
            </div>
            <div className="space-y-1 font-mono text-xs">
              <div className="flex justify-between text-[#94A3B8]">
                <span>Total Nodes:</span>
                <span className="text-[#F8FAFC]">{stats.splits.validation.total_nodes.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-[#94A3B8]">
                <span>Labeled Subset:</span>
                <span className="text-[#F8FAFC]">{stats.splits.validation.labeled_nodes.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-[#94A3B8]">
                <span>Illicit Nodes:</span>
                <span className="text-[#EF4444]">{stats.splits.validation.illicit_nodes.toLocaleString()}</span>
              </div>
            </div>
            <span className="text-[11px] font-mono text-[#64748B] block pt-1 border-t border-[#1E293B]">
              {stats.splits.validation.ratio}
            </span>
          </div>

          <div className="bg-[#080C14] border border-[#1E293B] rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase text-[#10B981] font-semibold">Test Evaluation Split</span>
              <span className="text-xs font-mono text-[#94A3B8]">Timesteps {stats.splits.test.timesteps}</span>
            </div>
            <div className="space-y-1 font-mono text-xs">
              <div className="flex justify-between text-[#94A3B8]">
                <span>Total Nodes:</span>
                <span className="text-[#F8FAFC]">{stats.splits.test.total_nodes.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-[#94A3B8]">
                <span>Labeled Subset:</span>
                <span className="text-[#F8FAFC]">{stats.splits.test.labeled_nodes.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-[#94A3B8]">
                <span>Illicit Nodes:</span>
                <span className="text-[#EF4444]">{stats.splits.test.illicit_nodes.toLocaleString()}</span>
              </div>
            </div>
            <span className="text-[11px] font-mono text-[#64748B] block pt-1 border-t border-[#1E293B]">
              {stats.splits.test.ratio}
            </span>
          </div>
        </div>
      </section>

      {/* Critical Data Leakage Audit */}
      <section className="bg-[#111827] border border-[#1E293B] rounded-xl p-6 space-y-4">
        <div className="flex items-center gap-2.5 pb-3 border-b border-[#1E293B]">
          <AlertTriangle className="w-5 h-5 text-[#F59E0B]" />
          <div>
            <h3 className="text-base font-semibold text-[#F8FAFC]">Data Leakage Audit & Compliance Guarantee</h3>
            <p className="text-xs text-[#94A3B8]">Why Random K-Fold Splits Falsify Blockchain AML Benchmarks</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-[#94A3B8] leading-relaxed">
          <div className="space-y-2">
            <h4 className="text-sm font-semibold text-[#EF4444]">The Flaw: Random Graph Cross-Validation</h4>
            <p>
              When a graph model is trained using standard randomized K-fold splits, transaction nodes from future blocks ($t + k$) bleed information back to predecessor transactions in the past ($t$). Because Bitcoin transactions exist in tightly connected multi-hop chains, random sampling places counterparties of test transactions inside the training set, causing catastrophic temporal lookahead leakage.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="text-sm font-semibold text-[#10B981]">The Defense: Strict Temporal Split Protocol</h4>
            <p>
              CryptoScope AI enforces forward temporal evaluation: models are trained strictly on historical timesteps ($1 \le t \le 34$), tuned on validation timesteps ($35 \le t \le 41$), and evaluated only on unseen future timesteps ($42 \le t \le 49$). While this reduces vanity accuracy metrics from ~99% to realistic operational levels (80–85%), it guarantees zero temporal leakage and honest real-world performance.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
