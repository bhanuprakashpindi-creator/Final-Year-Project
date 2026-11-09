/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Anti-Money Laundering Forensic Command Center
 * Root Application Component
 */

import React, { useEffect, useState } from 'react';
import { Navigation } from './components/Navigation';
import { BackendStatusModal } from './components/BackendStatusModal';
import { HomePage } from './pages/HomePage';
import { InvestigatePage } from './pages/InvestigatePage';
import { NetworkExplorerPage } from './pages/NetworkExplorerPage';
import { IntelligencePage } from './pages/IntelligencePage';
import { DatasetPage } from './pages/DatasetPage';
import { ResearchPage } from './pages/ResearchPage';
import { apiClient } from './services/apiClient';
import { BackendHealth, TelemetryMode } from './types/aml';

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('home');
  const [currentTxId, setCurrentTxId] = useState<string>('230425980');
  const [health, setHealth] = useState<BackendHealth>(apiClient.getHealth());
  const [telemetryMode, setTelemetryMode] = useState<TelemetryMode>(apiClient.getTelemetryMode());
  const [settingsOpen, setSettingsOpen] = useState<boolean>(false);

  useEffect(() => {
    // Subscribe to backend health and telemetry
    const unsubscribe = apiClient.subscribe((newHealth, newMode) => {
      setHealth(newHealth);
      setTelemetryMode(newMode);
    });

    // Start background health polling every 15s
    apiClient.startHealthPolling(15000);

    return () => {
      unsubscribe();
      apiClient.stopHealthPolling();
    };
  }, []);

  const handleAnalyzeTransaction = (txId: string) => {
    setCurrentTxId(txId);
    setActiveTab('investigate');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleExploreGraph = (txId: string) => {
    setCurrentTxId(txId);
    setActiveTab('network');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-[#080C14] text-[#DFE2EE] flex flex-col font-sans selection:bg-[#00F2FE]/20 selection:text-[#00F2FE]">
      {/* Top Continuous Navigation Bar */}
      <Navigation
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        telemetryMode={telemetryMode}
        health={health}
        onOpenSettings={() => setSettingsOpen(true)}
        onQuickSearch={handleAnalyzeTransaction}
      />

      {/* Main Viewport Container */}
      <main className="flex-1 w-full pb-16">
        {activeTab === 'home' && (
          <HomePage
            onAnalyzeTransaction={handleAnalyzeTransaction}
            onNavigateTab={setActiveTab}
          />
        )}

        {activeTab === 'investigate' && (
          <InvestigatePage
            currentTxId={currentTxId}
            onExploreGraph={handleExploreGraph}
          />
        )}

        {activeTab === 'network' && (
          <NetworkExplorerPage
            initialTxId={currentTxId}
            onInvestigateTx={handleAnalyzeTransaction}
          />
        )}

        {activeTab === 'intelligence' && <IntelligencePage />}

        {activeTab === 'dataset' && <DatasetPage />}

        {activeTab === 'research' && <ResearchPage />}
      </main>

      {/* Backend Status & Connectivity Modal */}
      <BackendStatusModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        health={health}
        telemetryMode={telemetryMode}
        onRefresh={() => {
          setHealth(apiClient.getHealth());
          setTelemetryMode(apiClient.getTelemetryMode());
        }}
      />
    </div>
  );
}
