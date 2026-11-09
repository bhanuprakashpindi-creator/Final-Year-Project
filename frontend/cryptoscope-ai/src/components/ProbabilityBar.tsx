/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Calibrated Animated Probability Bar
 */

import React from 'react';
import { motion } from 'motion/react';

interface ProbabilityBarProps {
  illicitProb: number;
  licitProb: number;
  showLabels?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export const ProbabilityBar: React.FC<ProbabilityBarProps> = ({
  illicitProb,
  licitProb,
  showLabels = true,
  size = 'md',
}) => {
  const illicitPercent = Math.min(Math.max(illicitProb * 100, 0), 100);
  const licitPercent = Math.min(Math.max(licitProb * 100, 0), 100);

  const heightClasses = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-3.5',
  };

  return (
    <div className="w-full space-y-1.5">
      {/* Numbers */}
      {showLabels && (
        <div className="flex items-center justify-between text-xs font-mono tabular-nums">
          <div className="flex items-center gap-1.5">
            <span className="text-[#EF4444] font-medium">Illicit</span>
            <span className="text-[#EF4444] font-semibold">{illicitPercent.toFixed(1)}%</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[#10B981] font-semibold">{licitPercent.toFixed(1)}%</span>
            <span className="text-[#10B981] font-medium">Licit</span>
          </div>
        </div>
      )}

      {/* Progress Bar Container */}
      <div className={`w-full ${heightClasses[size]} bg-[#111827] border border-[#1E293B] rounded-full overflow-hidden flex`}>
        {/* Illicit side */}
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${illicitPercent}%` }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="h-full bg-gradient-to-r from-[#EF4444] to-[#F87171]"
        />
        {/* Licit side */}
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${licitPercent}%` }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="h-full bg-gradient-to-r from-[#34D399] to-[#10B981]"
        />
      </div>
    </div>
  );
};
