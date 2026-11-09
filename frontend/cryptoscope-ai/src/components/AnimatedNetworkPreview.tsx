/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * CryptoScope AI — Animated Network Graph Preview (Homepage Hero)
 */

import React, { useEffect, useRef } from 'react';

export const AnimatedNetworkPreview: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = canvas.parentElement?.clientWidth || 600);
    let height = (canvas.height = canvas.parentElement?.clientHeight || 400);

    const handleResize = () => {
      if (!canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };

    window.addEventListener('resize', handleResize);

    // Realistic Elliptic transaction cluster structure
    interface Particle {
      x: number;
      y: number;
      vx: number;
      vy: number;
      radius: number;
      type: 'central' | 'illicit' | 'licit' | 'unknown';
      pulse: number;
    }

    const particles: Particle[] = [];
    const count = Math.min(Math.floor((width * height) / 9000), 45);

    for (let i = 0; i < count; i++) {
      let type: Particle['type'] = 'unknown';
      if (i === 0) type = 'central';
      else if (i % 5 === 0) type = 'illicit';
      else if (i % 2 === 0) type = 'licit';

      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        radius: type === 'central' ? 6 : (type === 'illicit' ? 4 : 3),
        type,
        pulse: Math.random() * Math.PI,
      });
    }

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw connecting edges
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 120) {
            const alpha = (1 - dist / 120) * 0.25;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = particles[i].type === 'illicit' || particles[j].type === 'illicit'
              ? `rgba(239, 68, 68, ${alpha})`
              : `rgba(0, 242, 254, ${alpha})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      // Draw and update particles
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        p.pulse += 0.03;

        // Bounce off bounds
        if (p.x < 10 || p.x > width - 10) p.vx *= -1;
        if (p.y < 10 || p.y > height - 10) p.vy *= -1;

        ctx.beginPath();
        const pulseRadius = p.radius + Math.sin(p.pulse) * 0.8;
        ctx.arc(p.x, p.y, Math.max(pulseRadius, 1), 0, Math.PI * 2);

        if (p.type === 'central') {
          ctx.fillStyle = '#00F2FE';
          ctx.shadowColor = '#00F2FE';
          ctx.shadowBlur = 10;
        } else if (p.type === 'illicit') {
          ctx.fillStyle = '#EF4444';
          ctx.shadowColor = '#EF4444';
          ctx.shadowBlur = 6;
        } else if (p.type === 'licit') {
          ctx.fillStyle = '#10B981';
          ctx.shadowColor = '#10B981';
          ctx.shadowBlur = 4;
        } else {
          ctx.fillStyle = '#64748B';
          ctx.shadowBlur = 0;
        }

        ctx.fill();
        ctx.shadowBlur = 0; // reset
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <div className="relative w-full h-full min-h-[340px] bg-gradient-to-b from-[#0B0F19] to-[#080C14] rounded-xl border border-[#1E293B] overflow-hidden flex items-center justify-center">
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
      <div className="absolute inset-0 bg-gradient-to-t from-[#080C14] via-transparent to-transparent pointer-events-none" />
      
      {/* Overlay telemetry pill */}
      <div className="absolute top-4 left-4 z-10 flex items-center gap-2 bg-[#0B0F19]/80 backdrop-blur-md px-3 py-1.5 border border-[#1E293B] rounded-md text-[11px] font-mono text-[#94A3B8]">
        <span className="w-2 h-2 rounded-full bg-[#00F2FE] animate-pulse" />
        <span>Topological Message Passing (234,355 Edges)</span>
      </div>
    </div>
  );
};
