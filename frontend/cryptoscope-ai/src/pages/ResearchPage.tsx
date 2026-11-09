/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Research Page (Academic Foundations & IEEE Access Grounding)
 */

import React from 'react';
import {
  BookOpen,
  ExternalLink,
  Award,
  Layers,
  Cpu,
  GitFork,
  Binary,
  Share2,
  CheckCircle2,
  AlertCircle,
  FileText,
} from 'lucide-react';
import { IEEE_PAPER_COMPARISON } from '../data/ellipticBenchmark';

export const ResearchPage: React.FC = () => {
  const paper = IEEE_PAPER_COMPARISON;

  return (
    <div className="space-y-12 py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="pb-4 border-b border-[#1E293B]">
        <span className="text-xs font-mono uppercase tracking-wider text-[#00F2FE]">Academic Foundation</span>
        <h1 className="text-2xl sm:text-3xl font-semibold text-[#F8FAFC] mt-1">
          Graph Neural Networks in Cryptocurrency AML
        </h1>
        <p className="text-xs sm:text-sm text-[#94A3B8] mt-1.5 max-w-2xl leading-relaxed">
          Grounding CryptoScope AI in peer-reviewed forensic literature, graph representation learning, and dynamic UTXO network analysis.
        </p>
      </div>

      {/* IEEE Access Paper Formal Reference */}
      <section className="bg-[#111827] border border-[#1E293B] rounded-xl p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-[#00F2FE]/10 text-[#00F2FE] border border-[#00F2FE]/30 uppercase font-semibold">
                IEEE Access Publication
              </span>
              <span className="text-xs font-mono text-[#64748B]">Volume 13 · 2025</span>
            </div>

            <h2 className="text-lg font-semibold text-[#F8FAFC] leading-snug">
              "{paper.citation.title}"
            </h2>

            <p className="text-xs text-[#94A3B8] font-mono">
              Authors: <strong className="text-[#F8FAFC]">{paper.citation.authors}</strong> ({paper.citation.year})
            </p>
            <p className="text-xs text-[#64748B]">
              Published in IEEE Access, Volume {paper.citation.volume}, pages {paper.citation.pages}.
            </p>
          </div>

          <a
            href={`https://doi.org/${paper.citation.doi}`}
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 bg-[#1E293B] hover:bg-[#334155] text-xs font-mono text-[#00F2FE] border border-[#334155] rounded-lg flex items-center gap-1.5 shrink-0 transition-colors shadow-sm"
          >
            <span>DOI: {paper.citation.doi}</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>

        <div className="p-3.5 bg-[#080C14] border border-[#1E293B] rounded-lg text-xs text-[#94A3B8] leading-relaxed">
          <strong className="text-[#F8FAFC]">Notice on Research Reproducibility:</strong> CryptoScope AI does not claim to perfectly replicate all stochastic parameters of the original IEEE Access study. Instead, we faithfully implement the mathematical layer formulations and compare our independently measured local results against the published benchmark ranges.
        </div>
      </section>

      {/* Research Motivation: Why GNNs for Crypto AML */}
      <section className="space-y-4">
        <div className="pb-2 border-b border-[#1E293B]">
          <h2 className="text-base font-semibold text-[#F8FAFC]">Why Cryptocurrency AML Demands Graph Analysis</h2>
          <p className="text-xs text-[#94A3B8]">The fundamental failure mode of tabular transaction surveillance.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 text-xs text-[#94A3B8] leading-relaxed">
          <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5 space-y-2">
            <h3 className="text-sm font-semibold text-[#F8FAFC]">Peeling Chains & Coin Dispersion</h3>
            <p>
              Launderers actively peel illicit funds by issuing consecutive multi-output transactions where small legitimate amounts are spent while the bulk change output is recursively forwarded. Tabular models observe each hop in isolation, completely missing the systemic multi-hop chain.
            </p>
          </div>

          <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5 space-y-2">
            <h3 className="text-sm font-semibold text-[#F8FAFC]">Mixer Hub-and-Spoke Topology</h3>
            <p>
              CoinJoin mixers aggregate hundreds of equalized inputs and outputs within single transactions. Graph Neural Networks detect the sudden convergence and divergence of topological flow vectors through message passing, even when fee and volume signals are obscured.
            </p>
          </div>

          <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5 space-y-2">
            <h3 className="text-sm font-semibold text-[#F8FAFC]">Topological Inductive Transfer</h3>
            <p>
              Because attackers constantly rotate Bitcoin wallet addresses, identity-based watchlists fail immediately. GNNs learn structural invariant embeddings of money laundering topologies, allowing them to classify transactions on unseen addresses.
            </p>
          </div>
        </div>
      </section>

      {/* Published IEEE Paper vs Our Local Measured Results */}
      <section className="bg-[#111827] border border-[#1E293B] rounded-xl p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-[#F8FAFC]">Published IEEE Results vs. Local Experimental Results</h2>
            <p className="text-xs text-[#94A3B8]">
              Side-by-side comparison of published paper performance ranges with our local measured test metrics.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-[#0B0F19] text-[#64748B] font-mono uppercase text-[10px] border-b border-[#1E293B]">
              <tr>
                <th className="py-3 px-3">Metric</th>
                <th className="py-3 px-3">Random Forest</th>
                <th className="py-3 px-3">GCN+Lin</th>
                <th className="py-3 px-3">GraphSAGE+Lin</th>
                <th className="py-3 px-3">GATv2+Lin</th>
                <th className="py-3 px-3">Chebyshev+Lin</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1E293B] font-mono">
              {paper.comparison_table.map((row, idx) => (
                <tr key={idx} className="hover:bg-[#1E293B]/40 transition-colors">
                  <td className="py-3 px-3 text-[#F8FAFC] font-semibold font-sans">{row.metric}</td>
                  <td className="py-3 px-3">
                    <span className="text-[#94A3B8] block text-[11px]">Pub: {row.random_forest.published}</span>
                    <span className="text-[#00F2FE] font-bold">Local: {row.random_forest.local_measured}</span>
                  </td>
                  <td className="py-3 px-3">
                    <span className="text-[#94A3B8] block text-[11px]">Pub: {row.gcn.published}</span>
                    <span className="text-[#00F2FE] font-bold">Local: {row.gcn.local_measured}</span>
                  </td>
                  <td className="py-3 px-3">
                    <span className="text-[#94A3B8] block text-[11px]">Pub: {row.graphsage.published}</span>
                    <span className="text-[#00F2FE] font-bold">Local: {row.graphsage.local_measured}</span>
                  </td>
                  <td className="py-3 px-3">
                    <span className="text-[#94A3B8] block text-[11px]">Pub: {row.gatv2.published}</span>
                    <span className="text-[#00F2FE] font-bold">Local: {row.gatv2.local_measured}</span>
                  </td>
                  <td className="py-3 px-3">
                    <span className="text-[#94A3B8] block text-[11px]">Pub: {row.chebyshev.published}</span>
                    <span className="text-[#00F2FE] font-bold">Local: {row.chebyshev.local_measured}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="text-xs text-[#94A3B8] pt-2 border-t border-[#1E293B] leading-relaxed">
          {paper.discussion}
        </p>
      </section>

      {/* Model Architectures & Mathematical Formulations */}
      <section className="space-y-4">
        <div className="pb-2 border-b border-[#1E293B]">
          <h2 className="text-base font-semibold text-[#F8FAFC]">Model Architecture Specifications</h2>
          <p className="text-xs text-[#94A3B8]">The mathematical mechanics powering the five classifiers.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5 space-y-2">
            <div className="flex items-center gap-2 text-sm font-semibold text-[#F8FAFC]">
              <Binary className="w-4 h-4 text-[#00F2FE]" />
              <span>Random Forest (Phase 2 Baseline)</span>
            </div>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Ensemble of 100 decorrelated decision trees. Each split optimizes Gini impurity over randomly sampled subsets of the 145 node features. Serves as the high-accuracy tabular benchmark against which topological GNN benefits are measured.
            </p>
          </div>

          <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5 space-y-2">
            <div className="flex items-center gap-2 text-sm font-semibold text-[#F8FAFC]">
              <Share2 className="w-4 h-4 text-[#00F2FE]" />
              <span>GCN+Lin (Spatial Message Passing)</span>
            </div>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Two-layer Graph Convolutional Network using Kipf & Welling normalization <code className="text-[#00F2FE] font-mono text-[11px]">D̃^(-1/2) Ã D̃^(-1/2) X Θ</code> augmented with a parallel linear skip transformation to retain original node features during aggregation.
            </p>
          </div>

          <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5 space-y-2">
            <div className="flex items-center gap-2 text-sm font-semibold text-[#F8FAFC]">
              <Layers className="w-4 h-4 text-[#00F2FE]" />
              <span>GraphSAGE+Lin (Inductive Neighborhood Aggregation)</span>
            </div>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Computes localized representations by sampling neighbor transactions and applying mean pooling followed by concatenation: <code className="text-[#00F2FE] font-mono text-[11px]">h_v^(k) = σ(W · [h_v^(k-1) || AGG(h_u^(k-1))])</code>.
            </p>
          </div>

          <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5 space-y-2">
            <div className="flex items-center gap-2 text-sm font-semibold text-[#F8FAFC]">
              <GitFork className="w-4 h-4 text-[#00F2FE]" />
              <span>GATv2+Lin (Dynamic Graph Attention)</span>
            </div>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Solves the static attention limitation of standard GAT by applying non-linear LeakyReLU activation prior to projection: <code className="text-[#00F2FE] font-mono text-[11px]">α_ij = softmax_j(a^T LeakyReLU(W · [h_i || h_j]))</code>. Employs 4 independent attention heads.
            </p>
          </div>
        </div>
      </section>

      {/* Limitations & Future Scope */}
      <section className="bg-[#111827] border border-[#1E293B] rounded-xl p-6 space-y-4">
        <h2 className="text-base font-semibold text-[#F8FAFC]">Limitations & Future Scope</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-[#94A3B8] leading-relaxed">
          <div className="p-3.5 bg-[#080C14] border border-[#1E293B] rounded-lg space-y-1.5">
            <strong className="text-[#F8FAFC] block">Unlabeled Dominance (77.1% Unknown)</strong>
            <p>
              Over 157,000 transactions in the Elliptic graph lack regulatory ground-truth labels. While GNNs leverage them for message-passing paths, developing semi-supervised contrastive learning objectives is vital to fully exploit unlabeled topology.
            </p>
          </div>

          <div className="p-3.5 bg-[#080C14] border border-[#1E293B] rounded-lg space-y-1.5">
            <strong className="text-[#F8FAFC] block">Dynamic Temporal Streaming</strong>
            <p>
              Current models evaluate on static snapshot timesteps. Future iterations will incorporate Temporal Graph Networks (TGN) and continuous-time memory banks to process real-time mempool transactions with sub-second latency.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
