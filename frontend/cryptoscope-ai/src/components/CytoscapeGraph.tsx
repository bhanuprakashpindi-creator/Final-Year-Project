/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Cytoscape.js Transaction Network Graph Component
 */

import React, { useEffect, useRef, useState } from 'react';
import cytoscape, { Core, EventObject } from 'cytoscape';
import { TransactionGraphData, GraphNode } from '../types/aml';
import { ZoomIn, ZoomOut, Maximize2, GitBranch, Layers, RotateCcw } from 'lucide-react';

interface CytoscapeGraphProps {
  graphData: TransactionGraphData;
  selectedNodeId: string | null;
  onSelectNode: (node: GraphNode) => void;
  hops: number;
  onToggleHops: (hops: number) => void;
}

export const CytoscapeGraph: React.FC<CytoscapeGraphProps> = ({
  graphData,
  selectedNodeId,
  onSelectNode,
  hops,
  onToggleHops,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [layoutName, setLayoutName] = useState<'concentric' | 'breadthfirst' | 'cose'>('concentric');
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Convert nodes and edges into Cytoscape format
    const cyNodes = graphData.nodes.map((node) => {
      let bgColor = '#64748B'; // unknown
      if (node.label === 'illicit' || node.predicted_class === 'illicit') {
        bgColor = '#EF4444';
      } else if (node.label === 'licit' || node.predicted_class === 'licit') {
        bgColor = '#10B981';
      }

      return {
        data: {
          id: node.id,
          label: node.is_central ? `TX: ${node.id.slice(0, 7)}...` : node.id.slice(0, 6),
          fullId: node.id,
          rawNode: node,
          bgColor,
          isCentral: node.is_central,
          hop: node.hop,
          size: node.is_central ? 48 : (node.hop === 1 ? 36 : 28),
        },
      };
    });

    const cyEdges = graphData.edges.map((edge) => ({
      data: {
        id: edge.id,
        source: edge.source,
        target: edge.target,
      },
    }));

    // Initialize Cytoscape
    const cy = cytoscape({
      container: containerRef.current,
      elements: [...cyNodes, ...cyEdges],
      boxSelectionEnabled: false,
      autounselectify: false,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(bgColor)',
            'label': 'data(label)',
            'color': '#F8FAFC',
            'font-family': 'JetBrains Mono, monospace',
            'font-size': '10px',
            'text-valign': 'bottom',
            'text-margin-y': 6,
            'text-background-color': '#080C14',
            'text-background-opacity': 0.85,
            'text-background-padding': '2px',
            'text-background-shape': 'roundrectangle',
            'width': 'data(size)',
            'height': 'data(size)',
            'border-width': 2,
            'border-color': '#1E293B',
            'transition-property': 'border-color, border-width, background-color',
            'transition-duration': 0.2,
          },
        },
        {
          selector: 'node[?isCentral]',
          style: {
            'border-width': 4,
            'border-color': '#00F2FE',
            'border-opacity': 0.9,
            'font-size': '11px',
            'font-weight': 'bold',
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': '#FFFFFF',
            'border-width': 4,
            'overlay-color': '#00F2FE',
            'overlay-opacity': 0.35,
            'overlay-padding': 4,
          },
        },
        {
          selector: 'edge',
          style: {
            'width': 1.8,
            'line-color': 'rgba(148, 163, 184, 0.35)',
            'target-arrow-color': 'rgba(148, 163, 184, 0.75)',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'arrow-scale': 1.2,
          },
        },
        {
          selector: 'edge:selected',
          style: {
            'line-color': '#00F2FE',
            'target-arrow-color': '#00F2FE',
            'width': 2.5,
          },
        },
      ],
      layout: getLayoutConfig(layoutName, graphData.central_node_id),
    });

    // Event Handlers
    cy.on('tap', 'node', (evt: EventObject) => {
      const raw = evt.target.data('rawNode') as GraphNode;
      if (raw) {
        onSelectNode(raw);
      }
    });

    cy.on('mouseover', 'node', (evt: EventObject) => {
      const raw = evt.target.data('rawNode') as GraphNode;
      if (raw) setHoveredNode(raw);
    });

    cy.on('mouseout', 'node', () => {
      setHoveredNode(null);
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [graphData, layoutName]);

  // Highlight selected node
  useEffect(() => {
    if (!cyRef.current || !selectedNodeId) return;
    cyRef.current.nodes().unselect();
    const target = cyRef.current.getElementById(selectedNodeId);
    if (target.length) {
      target.select();
    }
  }, [selectedNodeId]);

  function getLayoutConfig(name: string, rootId: string) {
    if (name === 'concentric') {
      return {
        name: 'concentric',
        concentric: (node: any) => {
          if (node.data('isCentral')) return 3;
          if (node.data('hop') === 1) return 2;
          return 1;
        },
        levelWidth: () => 1,
        padding: 50,
        animate: true,
        animationDuration: 400,
      };
    }
    if (name === 'breadthfirst') {
      return {
        name: 'breadthfirst',
        directed: true,
        roots: `#${rootId}`,
        padding: 40,
        spacingFactor: 1.25,
        animate: true,
      };
    }
    return {
      name: 'cose',
      idealEdgeLength: () => 90,
      nodeRepulsion: () => 4500,
      animate: true,
      animationDuration: 400,
    };
  }

  const handleZoomIn = () => cyRef.current?.zoom(cyRef.current.zoom() * 1.25);
  const handleZoomOut = () => cyRef.current?.zoom(cyRef.current.zoom() * 0.8);
  const handleFit = () => cyRef.current?.fit(undefined, 40);
  const handleReset = () => {
    if (cyRef.current) {
      cyRef.current.layout(getLayoutConfig(layoutName, graphData.central_node_id)).run();
      cyRef.current.fit(undefined, 40);
    }
  };

  return (
    <div className="relative w-full h-full min-h-[520px] bg-[#080C14] border border-[#1E293B] rounded-xl overflow-hidden flex flex-col">
      {/* Top Floating Toolbar */}
      <div className="absolute top-3 left-3 right-3 z-10 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
        {/* Left: Depth Selector and Layout */}
        <div className="flex items-center gap-1.5 pointer-events-auto bg-[#0B0F19]/90 backdrop-blur-md p-1 border border-[#1E293B] rounded-lg">
          <span className="text-[11px] font-mono text-[#64748B] px-2 uppercase">Depth:</span>
          <button
            onClick={() => onToggleHops(1)}
            className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
              hops === 1 ? 'bg-[#00F2FE] text-[#080C14] font-semibold' : 'text-[#94A3B8] hover:text-[#F8FAFC]'
            }`}
          >
            1-Hop
          </button>
          <button
            onClick={() => onToggleHops(2)}
            className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
              hops === 2 ? 'bg-[#00F2FE] text-[#080C14] font-semibold' : 'text-[#94A3B8] hover:text-[#F8FAFC]'
            }`}
          >
            2-Hop (Expand)
          </button>
        </div>

        {/* Right: Layout Switcher & Zoom Controls */}
        <div className="flex items-center gap-1 pointer-events-auto bg-[#0B0F19]/90 backdrop-blur-md p-1 border border-[#1E293B] rounded-lg">
          <button
            onClick={() => setLayoutName(layoutName === 'concentric' ? 'cose' : 'concentric')}
            title="Toggle Layout (Concentric / Force)"
            className="p-1.5 rounded text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-[#1E293B] text-xs font-mono flex items-center gap-1"
          >
            <GitBranch className="w-3.5 h-3.5 text-[#00F2FE]" />
            <span className="capitalize hidden sm:inline">{layoutName}</span>
          </button>
          <div className="w-[1px] h-4 bg-[#1E293B] mx-0.5" />
          <button
            onClick={handleZoomIn}
            title="Zoom In"
            className="p-1.5 rounded text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-[#1E293B]"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={handleZoomOut}
            title="Zoom Out"
            className="p-1.5 rounded text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-[#1E293B]"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={handleFit}
            title="Fit to Screen"
            className="p-1.5 rounded text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-[#1E293B]"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
          <button
            onClick={handleReset}
            title="Recenter Layout"
            className="p-1.5 rounded text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-[#1E293B]"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Canvas Viewport */}
      <div ref={containerRef} className="flex-1 w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Floating Bottom Legend */}
      <div className="absolute bottom-3 left-3 z-10 pointer-events-none flex items-center gap-3 bg-[#0B0F19]/90 backdrop-blur-md px-3 py-1.5 border border-[#1E293B] rounded-md text-[11px] font-mono">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[#EF4444]" />
          <span className="text-[#94A3B8]">Illicit</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[#10B981]" />
          <span className="text-[#94A3B8]">Licit</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[#64748B]" />
          <span className="text-[#94A3B8]">Unknown</span>
        </div>
        <div className="flex items-center gap-1.5 pl-1 border-l border-[#1E293B]">
          <span className="w-2.5 h-2.5 rounded-full border-2 border-[#00F2FE]" />
          <span className="text-[#00F2FE]">Target Node</span>
        </div>
      </div>

      {/* Hover Info Tooltip */}
      {hoveredNode && (
        <div className="absolute bottom-3 right-3 z-10 bg-[#0B0F19]/95 backdrop-blur-md border border-[#334155] p-2.5 rounded-lg text-xs font-mono space-y-1 pointer-events-none shadow-xl animate-fade-in">
          <div className="text-[#00F2FE] font-semibold">TxID: {hoveredNode.id}</div>
          <div className="text-[#94A3B8] flex items-center gap-2">
            <span>Label: <strong className="text-[#F8FAFC] uppercase">{hoveredNode.label}</strong></span>
            <span>·</span>
            <span>Hop: {hoveredNode.hop}</span>
          </div>
          <div className="text-[11px] text-[#64748B]">
            In-Degree: {hoveredNode.in_degree} · Out-Degree: {hoveredNode.out_degree}
          </div>
        </div>
      )}
    </div>
  );
};
