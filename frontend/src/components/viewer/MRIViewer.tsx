"use client";

/**
 * Interactive MRI Viewer Component.
 * Features: slice navigation, overlay toggle, opacity control, zoom/pan.
 * Uses canvas rendering for performance.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface MRIViewerProps {
  imageB64?: string;           // Base64 encoded slice image
  overlayB64?: string;         // Base64 encoded overlay
  gradcamB64?: string;         // Base64 encoded Grad-CAM
  totalSlices?: number;
  currentSlice?: number;
  onSliceChange?: (slice: number) => void;
  className?: string;
}

type ViewMode = "image" | "overlay" | "gradcam" | "split";

export default function MRIViewer({
  imageB64,
  overlayB64,
  gradcamB64,
  totalSlices = 1,
  currentSlice = 0,
  onSliceChange,
  className,
}: MRIViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<ViewMode>("overlay");
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [windowLevel, setWindowLevel] = useState({ center: 128, width: 256 });

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    const drawImage = (b64: string, alpha: number = 1) => {
      const img = new Image();
      img.onload = () => {
        ctx.globalAlpha = alpha;
        ctx.drawImage(img, 0, 0, canvas.width / zoom, canvas.height / zoom);
      };
      img.src = `data:image/png;base64,${b64}`;
    };

    if (mode === "image" && imageB64) {
      drawImage(imageB64);
    } else if (mode === "overlay") {
      if (imageB64) drawImage(imageB64, 1 - overlayOpacity);
      if (overlayB64) drawImage(overlayB64, overlayOpacity);
    } else if (mode === "gradcam" && gradcamB64) {
      drawImage(gradcamB64);
    } else if (mode === "split") {
      // Left half: image, right half: overlay
      if (imageB64) drawImage(imageB64);
      if (overlayB64) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(canvas.width / (2 * zoom), 0, canvas.width / zoom, canvas.height / zoom);
        ctx.clip();
        drawImage(overlayB64, 0.7);
        ctx.restore();
        // Draw divider line
        ctx.strokeStyle = "rgba(255,255,255,0.8)";
        ctx.lineWidth = 1 / zoom;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(canvas.width / (2 * zoom), 0);
        ctx.lineTo(canvas.width / (2 * zoom), canvas.height / zoom);
        ctx.stroke();
      }
    }

    ctx.restore();
  }, [imageB64, overlayB64, gradcamB64, mode, overlayOpacity, zoom, pan]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (e.ctrlKey) {
      // Zoom
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setZoom(z => Math.max(0.5, Math.min(5, z * delta)));
    } else {
      // Slice scroll
      const delta = e.deltaY > 0 ? 1 : -1;
      const newSlice = Math.max(0, Math.min(totalSlices - 1, currentSlice + delta));
      onSliceChange?.(newSlice);
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || e.altKey) {
      setIsPanning(true);
      setLastMouse({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isPanning) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    setPan(p => ({ x: p.x + dx, y: p.y + dy }));
    setLastMouse({ x: e.clientX, y: e.clientY });
  };

  const hasContent = imageB64 || overlayB64 || gradcamB64;

  return (
    <div className={cn("bg-gray-900 rounded-2xl overflow-hidden", className)}>
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-4 py-2 bg-gray-800 border-b border-gray-700">
        {/* View mode */}
        <div className="flex gap-1">
          {(["image","overlay","gradcam","split"] as ViewMode[]).map(m => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={cn(
                "px-2.5 py-1 rounded text-xs font-medium transition",
                mode === m
                  ? "bg-blue-600 text-white"
                  : "bg-gray-700 text-gray-300 hover:bg-gray-600"
              )}
            >
              {m === "overlay" ? "Seg" : m === "gradcam" ? "GradCAM" : m.charAt(0).toUpperCase() + m.slice(1)}
            </button>
          ))}
        </div>

        <div className="flex-1" />

        {/* Opacity slider */}
        {(mode === "overlay") && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400">Opacity</span>
            <input
              type="range" min={0} max={1} step={0.05}
              value={overlayOpacity}
              onChange={e => setOverlayOpacity(Number(e.target.value))}
              className="w-24 accent-blue-500"
            />
            <span className="text-xs text-gray-400 w-8">{Math.round(overlayOpacity * 100)}%</span>
          </div>
        )}

        {/* Zoom controls */}
        <div className="flex items-center gap-1">
          <button onClick={() => setZoom(z => Math.max(0.5, z - 0.2))}
            className="w-7 h-7 bg-gray-700 text-gray-300 rounded hover:bg-gray-600 text-sm flex items-center justify-center">−</button>
          <span className="text-xs text-gray-400 w-12 text-center">{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom(z => Math.min(5, z + 0.2))}
            className="w-7 h-7 bg-gray-700 text-gray-300 rounded hover:bg-gray-600 text-sm flex items-center justify-center">+</button>
          <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
            className="w-7 h-7 bg-gray-700 text-gray-300 rounded hover:bg-gray-600 text-xs flex items-center justify-center">⟳</button>
        </div>
      </div>

      {/* Canvas */}
      <div className="relative">
        {hasContent ? (
          <canvas
            ref={canvasRef}
            width={512}
            height={512}
            className="w-full cursor-crosshair"
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={() => setIsPanning(false)}
            onMouseLeave={() => setIsPanning(false)}
            style={{ imageRendering: "pixelated" }}
          />
        ) : (
          <div className="w-full h-64 flex items-center justify-center">
            <div className="text-center text-gray-500">
              <div className="text-4xl mb-2">🩻</div>
              <p className="text-sm">No image loaded</p>
            </div>
          </div>
        )}

        {/* Slice indicator */}
        {totalSlices > 1 && (
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black/60 text-white text-xs px-3 py-1 rounded-full">
            Slice {currentSlice + 1} / {totalSlices}
          </div>
        )}
      </div>

      {/* Slice slider */}
      {totalSlices > 1 && onSliceChange && (
        <div className="px-4 py-2 bg-gray-800 border-t border-gray-700">
          <input
            type="range" min={0} max={totalSlices - 1} value={currentSlice}
            onChange={e => onSliceChange(Number(e.target.value))}
            className="w-full accent-blue-500"
          />
        </div>
      )}
    </div>
  );
}
