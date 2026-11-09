/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Homepage (Google Stitch Design)
 */

import React, { useState } from 'react';
import {
  Search,
  ShieldCheck,
  Share2,
  Cpu,
  Layers,
  GitFork,
  Binary,
  ArrowRight,
  Database,
  ExternalLink,
  Lock,
} from 'lucide-react';
import { AnimatedNetworkPreview } from '../components/AnimatedNetworkPreview';
import { BENCHMARK_TRANSACTIONS } from '../data/ellipticBenchmark';

interface HomePageProps {
  onAnalyzeTransaction: (txId: string) => void;
  onNavigateTab: (tab: string) => void;
}

export const HomePage: React.FC<HomePageProps> = ({ onAnalyzeTransaction, onNavigateTab }) => {
  const [searchInput, setSearchInput] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const clean = searchInput.trim();
    if (!clean) {
      setError('Please enter an Elliptic transaction ID.');
      return;
    }
    setError(null);
    onAnalyzeTransaction(clean);
  };

  const handlePresetSelect = (txId: string) => {
    setSearchInput(txId);
    setError(null);
    onAnalyzeTransaction(txId);
  };

  const fiveModels = [
    {
      name: 'Random Forest',
      tag: 'Phase 2 Baseline',
      icon: Binary,
      description: 'Ensemble of 100 decorrelated decision trees evaluated on 145 local & aggregated node features.',
      metric: 'Illicit F1: 0.846',
    },
    {
      name: 'GCN+Lin',
      tag: 'Phase 3 Spatial GNN',
      icon: Share2,
      description: 'Kipf & Welling first-order spectral graph convolutional layers with residual linear transformations.',
      metric: 'Illicit F1: 0.804',
    },
    {
      name: 'GraphSAGE+Lin',
      tag: 'Phase 3 Inductive GNN',
      icon: Layers,
      description: 'Inductive mean neighborhood aggregation over sampled 1-hop and 2-hop transaction predecessors.',
      metric: 'Illicit F1: 0.837',
    },
    {
      name: 'GATv2+Lin',
      tag: 'Phase 3 Attention GNN',
      icon: GitFork,
      description: 'Dynamic multi-head self-attention weighting counterparty transaction flows adaptively.',
      metric: 'Illicit F1: 0.828',
    },
    {
      name: 'Chebyshev+Lin',
      tag: 'Phase 3 Spectral GNN',
      icon: Cpu,
      description: 'Chebyshev truncated polynomial filters capturing high-frequency localized graph Laplacian signals.',
      metric: 'Illicit F1: 0.792',
    },
  ];

  return (
    <div className="space-y-16 py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto animate-fade-in">
      {/* Hero Section */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center pt-4">
        {/* Left Column: Headlines & Omni Search */}
        <div className="lg:col-span-7 space-y-6">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 text-xs font-mono text-[#00F2FE] bg-[#00F2FE]/10 border border-[#00F2FE]/20 px-3 py-1 rounded-md">
              <span className="w-2 h-2 rounded-full bg-[#00F2FE] animate-pulse" />
              <span>Elliptic Dataset · 203,769 Transactions · 49 Timesteps</span>
            </div>

            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-semibold text-[#F8FAFC] tracking-tight leading-tight">
              Intelligence Beyond <span className="text-[#00F2FE]">Transactions.</span>
            </h1>

            <p className="text-sm sm:text-base text-[#94A3B8] max-w-xl leading-relaxed">
              Investigate cryptocurrency transaction networks using advanced machine learning and graph intelligence.
              Audit multi-hop UTXO flows, uncover laundering patterns, and run consensus inference across five models.
            </p>
          </div>

          {/* Primary Transaction Search Form */}
          <form onSubmit={handleSearch} className="space-y-2.5 max-w-lg">
            <div className="relative flex items-center">
              <input
                type="text"
                value={searchInput}
                onChange={(e) => {
                  setSearchInput(e.target.value);
                  if (error) setError(null);
                }}
                placeholder="Enter Elliptic Transaction ID (e.g. 230425980)..."
                className="w-full h-12 pl-11 pr-32 bg-[#111827] border border-[#1E293B] rounded-lg text-sm font-mono text-[#F8FAFC] placeholder-[#475569] focus:outline-none focus:border-[#00F2FE] focus:ring-1 focus:ring-[#00F2FE] shadow-inner transition-all"
              />
              <Search className="w-5 h-5 text-[#64748B] absolute left-3.5 pointer-events-none" />
              <button
                type="submit"
                className="absolute right-1.5 h-9 px-4 bg-[#00F2FE] hover:bg-[#38BDF8] text-[#080C14] text-xs font-semibold rounded-md flex items-center gap-1.5 transition-colors shadow-sm"
              >
                <span>Analyze</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

            {error && <p className="text-xs font-mono text-[#EF4444] pl-1">{error}</p>}

            {/* Quick-Pick Benchmark Presets */}
            <div className="pt-2 flex flex-wrap items-center gap-2 text-xs font-mono">
              <span className="text-[#64748B] text-[11px] uppercase tracking-wider">Verified Test Cases:</span>
              <button
                type="button"
                onClick={() => handlePresetSelect('230425980')}
                className="px-2 py-0.5 rounded bg-[#EF4444]/10 hover:bg-[#EF4444]/20 border border-[#EF4444]/30 text-[#EF4444] transition-colors"
              >
                #230425980 (Illicit)
              </button>
              <button
                type="button"
                onClick={() => handlePresetSelect('5530414')}
                className="px-2 py-0.5 rounded bg-[#10B981]/10 hover:bg-[#10B981]/20 border border-[#10B981]/30 text-[#10B981] transition-colors"
              >
                #5530414 (Licit)
              </button>
              <button
                type="button"
                onClick={() => handlePresetSelect('232022460')}
                className="px-2 py-0.5 rounded bg-[#F59E0B]/10 hover:bg-[#F59E0B]/20 border border-[#F59E0B]/30 text-[#F59E0B] transition-colors"
              >
                #232022460 (Mixer)
              </button>
              <button
                type="button"
                onClick={() => handlePresetSelect('232438397')}
                className="px-2 py-0.5 rounded bg-[#3B82F6]/10 hover:bg-[#3B82F6]/20 border border-[#3B82F6]/30 text-[#ADC6FF] transition-colors"
              >
                #232438397 (Unlabeled)
              </button>
            </div>
          </form>

          {/* Quick Metrics Bar */}
          <div className="pt-4 grid grid-cols-3 gap-3 border-t border-[#1E293B] max-w-lg">
            <div>
              <span className="text-[10px] font-mono uppercase tracking-wider text-[#64748B] block">Trained Models</span>
              <span className="text-base font-semibold text-[#F8FAFC] font-mono">5 Architectures</span>
            </div>
            <div>
              <span className="text-[10px] font-mono uppercase tracking-wider text-[#64748B] block">Graph Edges</span>
              <span className="text-base font-semibold text-[#F8FAFC] font-mono">234,355</span>
            </div>
            <div>
              <span className="text-[10px] font-mono uppercase tracking-wider text-[#64748B] block">Temporal Split</span>
              <span className="text-base font-semibold text-[#10B981] font-mono">Zero Leakage</span>
            </div>
          </div>
        </div>

        {/* Right Column: Animated Network Graph Canvas */}
        <div className="lg:col-span-5 h-[380px] sm:h-[420px]">
          <AnimatedNetworkPreview />
        </div>
      </section>

      {/* Five-Model Architecture Grid */}
      <section className="space-y-6 pt-4">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2 pb-3 border-b border-[#1E293B]">
          <div>
            <span className="text-xs font-mono uppercase tracking-wider text-[#00F2FE]">Multi-Architecture Inference</span>
            <h2 className="text-xl font-semibold text-[#F8FAFC] mt-0.5">The Five Forensic AML Models</h2>
          </div>
          <button
            onClick={() => onNavigateTab('intelligence')}
            className="text-xs font-mono text-[#00F2FE] hover:underline flex items-center gap-1 self-start sm:self-auto"
          >
            <span>View Benchmark Comparison</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          {fiveModels.map((m, idx) => {
            const Icon = m.icon;
            return (
              <div
                key={idx}
                className="bg-[#111827] border border-[#1E293B] hover:border-[#00F2FE]/40 rounded-xl p-4 flex flex-col justify-between transition-all group"
              >
                <div>
                  <div className="w-8 h-8 rounded-lg bg-[#1E293B] border border-[#334155]/60 flex items-center justify-center text-[#00F2FE] mb-3 group-hover:bg-[#00F2FE]/10 transition-colors">
                    <Icon className="w-4 h-4" />
                  </div>
                  <h3 className="text-sm font-semibold text-[#F8FAFC]">{m.name}</h3>
                  <span className="text-[10px] font-mono text-[#64748B] block mt-0.5 mb-2">{m.tag}</span>
                  <p className="text-xs text-[#94A3B8] leading-relaxed line-clamp-3">{m.description}</p>
                </div>
                <div className="mt-4 pt-2.5 border-t border-[#1E293B]">
                  <span className="text-xs font-mono text-[#00F2FE] font-medium">{m.metric}</span>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Four-Phase Research Journey */}
      <section className="space-y-6 pt-4">
        <div className="pb-3 border-b border-[#1E293B]">
          <span className="text-xs font-mono uppercase tracking-wider text-[#00F2FE]">Research Engineering Lifecycle</span>
          <h2 className="text-xl font-semibold text-[#F8FAFC] mt-0.5">Four Development Phases</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5 space-y-2">
            <span className="text-xs font-mono text-[#00F2FE]">Phase 1</span>
            <h4 className="text-sm font-semibold text-[#F8FAFC]">Dataset Engineering & Graph Prep</h4>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Standardized Elliptic node features into 145 reduced signals, constructed edge adjacency tensors, and built temporal split partitions (t1-34 train, t35-41 val, t42-49 test).
            </p>
          </div>

          <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5 space-y-2">
            <span className="text-xs font-mono text-[#00F2FE]">Phase 2</span>
            <h4 className="text-sm font-semibold text-[#F8FAFC]">Random Forest Baseline</h4>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Trained 100-tree ensemble on tabular node features with Gini impurity splitting to establish the non-graph baseline benchmark (Illicit F1: 0.846).
            </p>
          </div>

          <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5 space-y-2">
            <span className="text-xs font-mono text-[#00F2FE]">Phase 3</span>
            <h4 className="text-sm font-semibold text-[#F8FAFC]">Graph Neural Networks</h4>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Engineered and evaluated PyTorch Geometric models: GCN+Lin, GraphSAGE+Lin, GATv2+Lin with multi-head attention, and Chebyshev spectral filters.
            </p>
          </div>

          <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-5 space-y-2">
            <span className="text-xs font-mono text-[#00F2FE]">Phase 4</span>
            <h4 className="text-sm font-semibold text-[#F8FAFC]">Interactive Forensic Dashboard</h4>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Deployed CryptoScope AI platform with Cytoscape.js interactive network exploration, FastAPI integration, and multi-model consensus risk triage.
            </p>
          </div>
        </div>
      </section>

      {/* Academic Citation & Institutional Footer */}
      <footer className="pt-10 border-t border-[#1E293B] text-xs text-[#64748B] space-y-4">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-[#94A3B8] font-semibold">
              <ShieldCheck className="w-4 h-4 text-[#00F2FE]" />
              <span>CryptoScope AI · Final Year AML Research System</span>
            </div>
            <p className="text-[11px] text-[#64748B] mt-1">
              Based on the Elliptic Bitcoin Transaction Dataset and IEEE Access research (Ferretti et al., 2025, DOI: 10.1109/ACCESS.2025.3552240).
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <button
              onClick={() => onNavigateTab('research')}
              className="text-[#94A3B8] hover:text-[#00F2FE] transition-colors"
            >
              Research Paper
            </button>
            <span>·</span>
            <button
              onClick={() => onNavigateTab('dataset')}
              className="text-[#94A3B8] hover:text-[#00F2FE] transition-colors"
            >
              Dataset Audit
            </button>
            <span>·</span>
            <button
              onClick={() => onNavigateTab('intelligence')}
              className="text-[#94A3B8] hover:text-[#00F2FE] transition-colors"
            >
              Model Benchmarks
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
};
