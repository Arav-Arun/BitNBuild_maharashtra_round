"use client";
import { useEffect, useRef } from "react";

type Direction = "right" | "left" | "up" | "down" | "diagonal";

type ShapeGridProps = {
  direction?: Direction;
  speed?: number;
  squareSize?: number;
  borderColor?: string;
  fillColor?: string;
  hoverFillColor?: string;
  hoverTrailAmount?: number;
  className?: string;
};

// Animated square-grid canvas, modelled on the react-bits ShapeGrid component.
// Pointer tracking is on `window` so the grid can sit behind the page content.
export function ShapeGrid({
  direction = "right",
  speed = 0.4,
  squareSize = 44,
  borderColor = "#000",
  fillColor = "#fff",
  hoverFillColor = "#ececec",
  hoverTrailAmount = 0,
  className = "shapegrid-bg",
}: ShapeGridProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const step = reduceMotion ? 0 : speed;
    let width = 0;
    let height = 0;
    let offsetX = 0;
    let offsetY = 0;
    let hovered: { x: number; y: number } | null = null;
    let trail: { x: number; y: number }[] = [];
    let frame = 0;

    const resize = () => {
      const ratio = window.devicePixelRatio || 1;
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    };

    const onMove = (event: PointerEvent) => {
      const col = Math.floor((event.clientX - offsetX) / squareSize);
      const row = Math.floor((event.clientY - offsetY) / squareSize);
      if (hovered && hovered.x === col && hovered.y === row) return;
      if (hovered && hoverTrailAmount > 0) {
        trail = [hovered, ...trail].slice(0, hoverTrailAmount);
      }
      hovered = { x: col, y: row };
    };
    const onLeave = () => { hovered = null; };

    const draw = () => {
      const dx = direction === "right" ? 1 : direction === "left" ? -1 : direction === "diagonal" ? 1 : 0;
      const dy = direction === "down" ? 1 : direction === "up" ? -1 : direction === "diagonal" ? 1 : 0;
      offsetX = (((offsetX + dx * step) % squareSize) + squareSize) % squareSize;
      offsetY = (((offsetY + dy * step) % squareSize) + squareSize) % squareSize;

      ctx.fillStyle = fillColor;
      ctx.fillRect(0, 0, width, height);

      // Convert pointer cell back to screen space, since the grid keeps moving.
      const startX = offsetX - squareSize;
      const startY = offsetY - squareSize;
      const cols = Math.ceil(width / squareSize) + 2;
      const rows = Math.ceil(height / squareSize) + 2;
      const cellAt = (cell: { x: number; y: number }) => ({
        x: cell.x * squareSize + offsetX,
        y: cell.y * squareSize + offsetY,
      });

      ctx.fillStyle = hoverFillColor;
      trail.forEach((cell, index) => {
        const pos = cellAt(cell);
        ctx.globalAlpha = 0.6 * (1 - index / (trail.length + 1));
        ctx.fillRect(pos.x, pos.y, squareSize, squareSize);
      });
      ctx.globalAlpha = 1;
      if (hovered) {
        const pos = cellAt(hovered);
        ctx.fillRect(pos.x, pos.y, squareSize, squareSize);
      }

      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let c = 0; c < cols; c++) {
        const x = Math.round(startX + c * squareSize) + 0.5;
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }
      for (let r = 0; r < rows; r++) {
        const y = Math.round(startY + r * squareSize) + 0.5;
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }
      ctx.stroke();

      frame = requestAnimationFrame(draw);
    };

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerdown", onMove);
    document.addEventListener("pointerleave", onLeave);
    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerdown", onMove);
      document.removeEventListener("pointerleave", onLeave);
    };
  }, [direction, speed, squareSize, borderColor, fillColor, hoverFillColor, hoverTrailAmount]);

  return <canvas ref={canvasRef} className={className} aria-hidden="true" />;
}
