/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Horizontal Navigation Bar (Google Stitch Design)
 */

import React, { useState } from 'react';
import {
  ShieldAlert,
  Search,
  Share2,
  BarChart3,
  Database,
  BookOpen,
  Wifi,
  WifiOff,
  Settings,
  Menu,
  X,
  Radio,
} from 'lucide-react';
import { BackendHealth, TelemetryMode } from '../types/aml';

interface NavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  telemetryMode: TelemetryMode;
  health: BackendHealth;
  onOpenSettings: () => void;
  onQuickSearch: (txId: string) => void;
}

export const Navigation: React.FC<NavigationProps> = ({
  activeTab,
  setActiveTab,
  telemetryMode,
  health,
  onOpenSettings,
  onQuickSearch,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const navLinks = [
    { id: 'home', label: 'Home', icon: ShieldAlert },
    { id: 'investigate', label: 'Investigate', icon: Search },
    { id: 'network', label: 'Network Explorer', icon: Share2 },
    { id: 'intelligence', label: 'Intelligence', icon: BarChart3 },
    { id: 'dataset', label: 'Dataset', icon: Database },
    { id: 'research', label: 'Research', icon: BookOpen },
  ];

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      onQuickSearch(searchQuery.trim());
      setSearchQuery('');
    }
  };

  return (
    <header className="sticky top-0 z-50 h-14 w-full bg-[#0B0F19]/90 backdrop-blur-md border-b border-[#1E293B] px-4 lg:px-8">
      <div className="max-w-7xl mx-auto h-full flex items-center justify-between gap-4">
        {/* Brand Zone */}
        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={() => setActiveTab('home')}
            className="flex items-center gap-2.5 text-left group transition-opacity hover:opacity-90"
          >
            <div className="w-8 h-8 rounded-md bg-[#111827] border border-[#1E293B] flex items-center justify-center text-[#00F2FE] group-hover:border-[#00F2FE]/40 transition-colors">
              <ShieldAlert className="w-5 h-5 text-[#00F2FE]" />
            </div>
            <div>
              <span className="text-base font-semibold text-[#F8FAFC] tracking-tight">
                CryptoScope<span className="text-[#00F2FE]">.AI</span>
              </span>
            </div>
          </button>
        </div>

        {/* Center Navigation Links (Desktop) */}
        <nav className="hidden md:flex items-center gap-1 lg:gap-2">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const isActive = activeTab === link.id;
            return (
              <button
                key={link.id}
                onClick={() => setActiveTab(link.id)}
                className={`relative px-3 py-1.5 text-xs font-medium rounded-md transition-all flex items-center gap-1.5 whitespace-nowrap ${
                  isActive
                    ? 'text-[#00F2FE] bg-[#00F2FE]/10'
                    : 'text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-[#1E293B]/60'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-[#00F2FE]' : 'text-[#64748B]'}`} />
                <span>{link.label}</span>
                {isActive && (
                  <span className="absolute bottom-0 left-2 right-2 h-[2px] bg-[#00F2FE] rounded-full" />
                )}
              </button>
            );
          })}
        </nav>

        {/* Right Zone: Omni Search & Telemetry Status */}
        <div className="flex items-center gap-2.5 shrink-0">
          {/* Omni Search Mini */}
          <form onSubmit={handleSearchSubmit} className="hidden lg:block relative w-48 xl:w-56">
            <input
              type="text"
              placeholder="Search Elliptic TxID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full h-8 pl-7 pr-3 bg-[#111827] border border-[#1E293B] rounded-md text-xs font-mono text-[#F8FAFC] placeholder-[#475569] focus:outline-none focus:border-[#00F2FE] focus:ring-1 focus:ring-[#00F2FE]"
            />
            <Search className="w-3.5 h-3.5 text-[#64748B] absolute left-2 top-2.25 pointer-events-none" />
          </form>

          {/* Telemetry Status Trigger */}
          <button
            onClick={onOpenSettings}
            title={`Backend Status: ${health.status.toUpperCase()} (${telemetryMode}) - Click to configure`}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono border border-[#1E293B] bg-[#111827] hover:border-[#334155] transition-colors"
          >
            {health.status === 'online' ? (
              <>
                <span className="w-2 h-2 rounded-full bg-[#10B981] animate-pulse" />
                <span className="text-[#10B981] font-semibold hidden sm:inline">LIVE</span>
                {health.latency_ms && (
                  <span className="text-[#64748B] text-[10px] hidden xl:inline">({health.latency_ms}ms)</span>
                )}
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-[#3B82F6]" />
                <span className="text-[#ADC6FF] hidden sm:inline">BENCHMARK</span>
                <span className="text-[#64748B] text-[10px] hidden xl:inline">(Offline API)</span>
              </>
            )}
            <Settings className="w-3 h-3 text-[#64748B] ml-0.5" />
          </button>

          {/* Mobile menu hamburger */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-1.5 rounded-md text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-[#1E293B]"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-[#0B0F19] border-b border-[#1E293B] px-4 py-3 space-y-1">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const isActive = activeTab === link.id;
            return (
              <button
                key={link.id}
                onClick={() => {
                  setActiveTab(link.id);
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive ? 'bg-[#00F2FE]/10 text-[#00F2FE]' : 'text-[#94A3B8] hover:bg-[#1E293B]'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{link.label}</span>
              </button>
            );
          })}
        </div>
      )}
    </header>
  );
};
