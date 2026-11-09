/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Network Explorer Page (Cytoscape.js Directed Graph)
 */

import React, { useEffect, useState } from 'react';
import {
  Share2,
  Search,
  Layers,
  ArrowRight,
  AlertTriangle,
  RefreshCw,
  GitFork,
  Database,
  Info,
} from 'lucide-react';
import { GraphNode, TransactionGraphData } from '../types/aml';
import { apiClient } from '../services/apiClient';
import { CytoscapeGraph } from '../components/CytoscapeGraph';
import { NodeInspectorDrawer } from '../components/NodeInspectorDrawer';

interface NetworkExplorerPageProps {
  initialTxId: string;
  onInvestigateTx: (txId: string) => void;
}

export const NetworkExplorerPage: React.FC<NetworkExplorerPageProps> = ({
  initialTxId,
  onInvestigateTx,
}) => {
  const [currentId, setCurrentId] = useState(initialTxId || '230425980');
  const [inputVal, setInputVal] = useState(initialTxId || '230425980');
  const [hops, setHops] = useState<number>(1);
  const [graphData, setGraphData] = useState<TransactionGraphData | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchGraph = async (id: string, depth: number) => {
    if (!id.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.getNeighborhood(id.trim(), depth);
      setGraphData(data);
      // Auto-select central node
      const central = data.nodes.find((n) => n.is_central) || data.nodes[0] || null;
      setSelectedNode(central);
    } catch (err: any) {
      setGraphData(null);
      setError(err.message || 'Failed to load transaction graph.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (initialTxId) {
      setCurrentId(initialTxId);
      setInputVal(initialTxId);
      fetchGraph(initialTxId, hops);
    } else {
      fetchGraph('230425980', hops);
    }
  }, [initialTxId]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputVal.trim()) {
      setCurrentId(inputVal.trim());
      fetchGraph(inputVal.trim(), hops);
    }
  };

  const handleToggleHops = (newHops: number) => {
    setHops(newHops);
    fetchGraph(currentId, newHops);
  };

  return (
    <div className="space-y-6 py-6 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto animate-fade-in">
      {/* Top Search & Presets Filter Strip */}
      <section className="bg-[#111827] border border-[#1E293B] rounded-xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Search Input */}
        <form onSubmit={handleSearch} className="flex items-center gap-2 flex-1 max-w-md">
          <div className="relative flex-1">
            <input
              type="text"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              placeholder="Search Elliptic Transaction ID..."
              className="w-full h-9 pl-9 pr-3 bg-[#080C14] border border-[#1E293B] rounded-lg text-xs font-mono text-[#F8FAFC] placeholder-[#475569] focus:outline-none focus:border-[#00F2FE] focus:ring-1 focus:ring-[#00F2FE]"
            />
            <Search className="w-3.5 h-3.5 text-[#64748B] absolute left-3 top-3 pointer-events-none" />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="px-3.5 h-9 bg-[#00F2FE] hover:bg-[#38BDF8] text-[#080C14] text-xs font-semibold rounded-lg flex items-center gap-1 transition-colors shrink-0 disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
            <span>Load</span>
          </button>
        </form>

        {/* Quick Presets */}
        <div className="flex items-center gap-1.5 flex-wrap text-xs font-mono">
          <span className="text-[#64748B] text-[11px] uppercase mr-1">Presets:</span>
          <button
            onClick={() => {
              setInputVal('230425980');
              setCurrentId('230425980');
              fetchGraph('230425980', hops);
            }}
            className="px-2 py-1 rounded bg-[#EF4444]/10 hover:bg-[#EF4444]/20 text-[#EF4444] border border-[#EF4444]/30 transition-colors"
          >
            #230425980 (Illicit)
          </button>
          <button
            onClick={() => {
              setInputVal('5530414');
              setCurrentId('5530414');
              fetchGraph('5530414', hops);
            }}
            className="px-2 py-1 rounded bg-[#10B981]/10 hover:bg-[#10B981]/20 text-[#10B981] border border-[#10B981]/30 transition-colors"
          >
            #5530414 (Licit)
          </button>
          <button
            onClick={() => {
              setInputVal('232022460');
              setCurrentId('232022460');
              fetchGraph('232022460', hops);
            }}
            className="px-2 py-1 rounded bg-[#F59E0B]/10 hover:bg-[#F59E0B]/20 text-[#F59E0B] border border-[#F59E0B]/30 transition-colors"
          >
            #232022460 (Mixer)
          </button>
        </div>
      </section>

      {/* Graph Metrics & Status Bar */}
      {graphData && (
        <div className="bg-[#111827] border border-[#1E293B] rounded-xl px-4 py-3 flex flex-wrap items-center justify-between gap-4 text-xs font-mono">
          <div className="flex items-center gap-6">
            <div>
              <span className="text-[#64748B] text-[10px] uppercase block">Subnetwork Nodes</span>
              <span className="text-sm font-bold text-[#F8FAFC]">{graphData.stats.total_nodes}</span>
            </div>
            <div>
              <span className="text-[#64748B] text-[10px] uppercase block">Directed Edges</span>
              <span className="text-sm font-bold text-[#F8FAFC]">{graphData.stats.total_edges}</span>
            </div>
            <div>
              <span className="text-[#64748B] text-[10px] uppercase block">Direct Inflows</span>
              <span className="text-sm font-bold text-[#3B82F6]">{graphData.stats.inflows}</span>
            </div>
            <div>
              <span className="text-[#64748B] text-[10px] uppercase block">Direct Outflows</span>
              <span className="text-sm font-bold text-[#3B82F6]">{graphData.stats.outflows}</span>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#EF4444]" />
              <span className="text-[#94A3B8]">Illicit: <strong className="text-[#F8FAFC]">{graphData.stats.illicit_count}</strong></span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#10B981]" />
              <span className="text-[#94A3B8]">Licit: <strong className="text-[#F8FAFC]">{graphData.stats.licit_count}</strong></span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#64748B]" />
              <span className="text-[#94A3B8]">Unknown: <strong className="text-[#F8FAFC]">{graphData.stats.unknown_count}</strong></span>
            </div>
            <span
              className={`text-[10px] px-2 py-0.5 rounded border uppercase ml-2 ${
                graphData.provenance === 'live_model_inference'
                  ? 'bg-[#10B981]/15 text-[#10B981] border-[#10B981]/40'
                  : 'bg-[#3B82F6]/15 text-[#ADC6FF] border-[#3B82F6]/40'
              }`}
            >
              {graphData.provenance === 'live_model_inference' ? '🟢 LIVE GRAPH' : '🔵 VERIFIED SUBGRAPH'}
            </span>
          </div>
        </div>
      )}

      {/* Error Notice */}
      {error && !loading && (
        <div className="p-6 bg-[#EF4444]/10 border border-[#EF4444]/30 rounded-xl space-y-2 text-xs">
          <div className="flex items-center gap-2 text-[#EF4444] font-semibold text-sm">
            <AlertTriangle className="w-5 h-5" />
            <span>Graph Extraction Error</span>
          </div>
          <p className="text-[#F8FAFC] leading-relaxed">{error}</p>
          <p className="text-[#94A3B8]">
            Note: As per data authenticity guidelines, synthetic edges are never generated.
            Please connect the Python backend or select a verified benchmark transaction.
          </p>
        </div>
      )}

      {/* Graph Canvas & Side Inspector Drawer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[580px]">
        {/* Cytoscape Canvas */}
        <div className={`${selectedNode ? 'lg:col-span-8 xl:col-span-9' : 'lg:col-span-12'} h-[580px]`}>
          {graphData ? (
            <CytoscapeGraph
              graphData={graphData}
              selectedNodeId={selectedNode ? selectedNode.id : null}
              onSelectNode={(node) => setSelectedNode(node)}
              hops={hops}
              onToggleHops={handleToggleHops}
            />
          ) : (
            <div className="w-full h-full bg-[#080C14] border border-[#1E293B] rounded-xl flex items-center justify-center p-8 text-center text-xs text-[#94A3B8]">
              {loading ? (
                <div className="space-y-2">
                  <RefreshCw className="w-6 h-6 text-[#00F2FE] animate-spin mx-auto" />
                  <p>Extracting directed graph neighborhood...</p>
                </div>
              ) : (
                <p>No transaction graph loaded. Please enter a transaction ID or select a preset.</p>
              )}
            </div>
          )}
        </div>

        {/* Selected Node Forensic Inspector Drawer */}
        {selectedNode && (
          <div className="lg:col-span-4 xl:col-span-3">
            <NodeInspectorDrawer
              node={selectedNode}
              onClose={() => setSelectedNode(null)}
              onInvestigateNode={(txId) => onInvestigateTx(txId)}
            />
          </div>
        )}
      </div>
    </div>
  );
};
