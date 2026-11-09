/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Node Forensic Inspector Drawer Component
 */

import React, { useState } from 'react';
import { GraphNode } from '../types/aml';
import {
  X,
  Copy,
  Check,
  Search,
  ArrowDownLeft,
  ArrowUpRight,
  Clock,
  Layers,
  Shield,
  Activity,
} from 'lucide-react';

interface NodeInspectorDrawerProps {
  node: GraphNode | null;
  onClose: () => void;
  onInvestigateNode: (txId: string) => void;
}

export const NodeInspectorDrawer: React.FC<NodeInspectorDrawerProps> = ({
  node,
  onClose,
  onInvestigateNode,
}) => {
  const [copied, setCopied] = useState(false);

  if (!node) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(node.id);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isIllicit = node.label === 'illicit' || node.predicted_class === 'illicit';
  const isLicit = node.label === 'licit' || node.predicted_class === 'licit';

  return (
    <div className="w-full lg:w-96 bg-[#0B0F19] border border-[#1E293B] rounded-xl p-5 flex flex-col justify-between shadow-2xl animate-fade-in space-y-4">
      {/* Top Header */}
      <div>
        <div className="flex items-start justify-between gap-2 pb-3 border-b border-[#1E293B]">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono uppercase tracking-wider text-[#00F2FE] bg-[#00F2FE]/10 px-1.5 py-0.5 rounded">
                {node.is_central ? 'Central Focus Node' : `${node.hop}-Hop Counterparty`}
              </span>
            </div>
            <h3 className="text-sm font-semibold text-[#F8FAFC] mt-1.5 flex items-center gap-1.5 font-mono">
              <span>{node.id}</span>
              <button
                onClick={handleCopy}
                title="Copy Transaction ID"
                className="p-1 rounded text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-[#1E293B] transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-[#10B981]" /> : <Copy className="w-3.5 h-3.5" />}
              </button>
            </h3>
          </div>

          <button
            onClick={onClose}
            className="p-1 rounded-md text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-[#1E293B]"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Status Breakdown */}
        <div className="my-4 space-y-3">
          {/* Ground Truth vs Prediction */}
          <div className="bg-[#111827] border border-[#1E293B] rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-[#94A3B8]">Dataset Ground Truth</span>
              <span
                className={`font-mono text-[11px] font-semibold uppercase px-2 py-0.5 rounded ${
                  node.label === 'illicit'
                    ? 'bg-[#EF4444]/15 text-[#EF4444] border border-[#EF4444]/30'
                    : node.label === 'licit'
                    ? 'bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/30'
                    : 'bg-[#64748B]/15 text-[#94A3B8] border border-[#64748B]/30'
                }`}
              >
                {node.label}
              </span>
            </div>

            {node.predicted_class && (
              <div className="flex items-center justify-between text-xs pt-2 border-t border-[#1E293B]">
                <span className="text-[#94A3B8]">Model Consensus</span>
                <span
                  className={`font-mono text-[11px] font-semibold uppercase px-2 py-0.5 rounded ${
                    isIllicit
                      ? 'bg-[#EF4444]/15 text-[#EF4444] border border-[#EF4444]/30'
                      : 'bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/30'
                  }`}
                >
                  {node.predicted_class}
                  {node.illicit_prob !== undefined && ` (${(node.illicit_prob * 100).toFixed(0)}%)`}
                </span>
              </div>
            )}
          </div>

          {/* Graph Metrics Grid */}
          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div className="p-2.5 bg-[#111827] border border-[#1E293B] rounded-lg">
              <div className="flex items-center gap-1.5 text-[#64748B] text-[11px] mb-1">
                <Clock className="w-3.5 h-3.5 text-[#00F2FE]" />
                <span>Timestep</span>
              </div>
              <span className="text-sm font-semibold text-[#F8FAFC]">t = {node.timestep}</span>
            </div>

            <div className="p-2.5 bg-[#111827] border border-[#1E293B] rounded-lg">
              <div className="flex items-center gap-1.5 text-[#64748B] text-[11px] mb-1">
                <Layers className="w-3.5 h-3.5 text-[#00F2FE]" />
                <span>Hop Distance</span>
              </div>
              <span className="text-sm font-semibold text-[#F8FAFC]">{node.hop}</span>
            </div>

            <div className="p-2.5 bg-[#111827] border border-[#1E293B] rounded-lg">
              <div className="flex items-center gap-1.5 text-[#64748B] text-[11px] mb-1">
                <ArrowDownLeft className="w-3.5 h-3.5 text-[#3B82F6]" />
                <span>In-Degree</span>
              </div>
              <span className="text-sm font-semibold text-[#F8FAFC]">{node.in_degree}</span>
            </div>

            <div className="p-2.5 bg-[#111827] border border-[#1E293B] rounded-lg">
              <div className="flex items-center gap-1.5 text-[#64748B] text-[11px] mb-1">
                <ArrowUpRight className="w-3.5 h-3.5 text-[#3B82F6]" />
                <span>Out-Degree</span>
              </div>
              <span className="text-sm font-semibold text-[#F8FAFC]">{node.out_degree}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Action Footer */}
      <div className="pt-3 border-t border-[#1E293B]">
        <button
          onClick={() => onInvestigateNode(node.id)}
          className="w-full py-2 px-3 bg-[#00F2FE] hover:bg-[#38BDF8] text-[#080C14] text-xs font-semibold rounded-md flex items-center justify-center gap-2 transition-colors shadow-sm"
        >
          <Search className="w-3.5 h-3.5" />
          <span>Investigate in Five Models</span>
        </button>
      </div>
    </div>
  );
};
