/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Model Prediction Card Component
 */

import React from 'react';
import { ModelPrediction } from '../types/aml';
import { ProbabilityBar } from './ProbabilityBar';
import { Network, Layers, GitFork, Cpu, Binary } from 'lucide-react';

interface ModelPredictionCardProps {
  prediction: ModelPrediction;
}

export const ModelPredictionCard: React.FC<ModelPredictionCardProps> = ({ prediction }) => {
  const isIllicit = prediction.predicted_class === 'illicit';

  const getModelIcon = () => {
    switch (prediction.model_id) {
      case 'random_forest':
        return Binary;
      case 'gcn':
        return Network;
      case 'graphsage':
        return Layers;
      case 'gatv2':
        return GitFork;
      case 'chebyshev':
        return Cpu;
      default:
        return Network;
    }
  };

  const Icon = getModelIcon();

  return (
    <div className="bg-[#111827] border border-[#1E293B] hover:border-[#334155] rounded-xl p-4 transition-all duration-200">
      {/* Top Header */}
      <div className="flex items-start justify-between gap-2 mb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-[#1E293B] border border-[#334155]/60 flex items-center justify-center text-[#00F2FE]">
            <Icon className="w-4 h-4" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-[#F8FAFC] tracking-tight">{prediction.model_name}</h4>
            <span className="text-[11px] font-mono text-[#64748B] block">{prediction.model_type}</span>
          </div>
        </div>

        <div className="text-right">
          <span
            className={`inline-block px-2 py-0.5 rounded text-[11px] font-mono font-semibold uppercase tracking-wider ${
              isIllicit
                ? 'bg-[#EF4444]/15 text-[#EF4444] border border-[#EF4444]/30'
                : 'bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/30'
            }`}
          >
            {prediction.predicted_class}
          </span>
        </div>
      </div>

      {/* Animated Probability Bar */}
      <div className="my-3 pt-1">
        <ProbabilityBar illicitProb={prediction.illicit_prob} licitProb={prediction.licit_prob} size="sm" />
      </div>

      {/* Confidence Footer */}
      <div className="pt-2 border-t border-[#1E293B] flex items-center justify-between text-xs text-[#94A3B8]">
        <span className="text-[11px] text-[#64748B]">Model Confidence</span>
        <span className="font-mono tabular-nums text-[#F8FAFC] font-medium">
          {(prediction.confidence * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
};
