"use client";

import { useState } from "react";

function escapeToken(token: string): string {
  return token.replace(/~/g, "~0").replace(/\//g, "~1");
}

export function childPointer(parent: string, key: string | number): string {
  return `${parent}/${escapeToken(String(key))}`;
}

interface JsonViewProps {
  value: unknown;
  /** JSON pointer of the root value (empty string = document root). */
  rootPointer?: string;
  highlight?: string | null;
  onValueClick?: (pointer: string, value: unknown) => void;
  collapseAfter?: number;
}

/**
 * Renders JSON with every scalar addressable by its JSON pointer. When onValueClick is
 * provided, scalars become buttons so a reader can ask where a value came from.
 */
export function JsonView({
  value,
  rootPointer = "",
  highlight,
  onValueClick,
  collapseAfter = 40,
}: JsonViewProps) {
  return (
    <div className="json">
      <Node
        value={value}
        pointer={rootPointer}
        depth={0}
        highlight={highlight ?? null}
        onValueClick={onValueClick}
        collapseAfter={collapseAfter}
      />
    </div>
  );
}

interface NodeProps {
  value: unknown;
  pointer: string;
  depth: number;
  highlight: string | null;
  onValueClick?: (pointer: string, value: unknown) => void;
  collapseAfter: number;
}

function Node({ value, pointer, depth, highlight, onValueClick, collapseAfter }: NodeProps) {
  const [open, setOpen] = useState(true);
  const indent = "  ".repeat(depth + 1);
  const closeIndent = "  ".repeat(depth);

  if (value === null || typeof value !== "object") {
    return <Scalar value={value} pointer={pointer} highlight={highlight} onValueClick={onValueClick} />;
  }

  const isArray = Array.isArray(value);
  const entries: [string | number, unknown][] = isArray
    ? (value as unknown[]).map((v, i) => [i, v])
    : Object.entries(value as Record<string, unknown>);
  const [openChar, closeChar] = isArray ? ["[", "]"] : ["{", "}"];

  if (entries.length === 0) {
    return <span className="faint">{openChar + closeChar}</span>;
  }

  if (!open) {
    return (
      <button type="button" className="value-link faint" onClick={() => setOpen(true)}>
        {openChar} {entries.length} {isArray ? "items" : "keys"} {closeChar}
      </button>
    );
  }

  const visible = entries.slice(0, collapseAfter);
  return (
    <span>
      <button
        type="button"
        className="faint"
        style={{ background: "none", border: 0, cursor: "pointer", padding: 0 }}
        onClick={() => setOpen(false)}
        aria-label="Collapse"
      >
        {openChar}
      </button>
      {"\n"}
      {visible.map(([key, child], index) => {
        const childPtr = childPointer(pointer, key);
        return (
          <span key={String(key)}>
            {indent}
            {!isArray && <span style={{ color: "var(--text-2)" }}>{JSON.stringify(key)}</span>}
            {!isArray && ": "}
            <Node
              value={child}
              pointer={childPtr}
              depth={depth + 1}
              highlight={highlight}
              onValueClick={onValueClick}
              collapseAfter={collapseAfter}
            />
            {index < entries.length - 1 ? "," : ""}
            {"\n"}
          </span>
        );
      })}
      {entries.length > visible.length && (
        <span className="faint">
          {indent}… {entries.length - visible.length} more{"\n"}
        </span>
      )}
      {closeIndent}
      {closeChar}
    </span>
  );
}

function Scalar({
  value,
  pointer,
  highlight,
  onValueClick,
}: {
  value: unknown;
  pointer: string;
  highlight: string | null;
  onValueClick?: (pointer: string, value: unknown) => void;
}) {
  const text = typeof value === "string" ? JSON.stringify(value) : String(value);
  const color =
    typeof value === "string"
      ? "#c3e88d"
      : typeof value === "number"
        ? "#f78c6c"
        : typeof value === "boolean"
          ? "#c792ea"
          : "var(--text-3)";
  const isHighlighted = highlight !== null && highlight === pointer;
  if (!onValueClick || value === null) {
    return (
      <span className={isHighlighted ? "highlight" : undefined} style={{ color }}>
        {text}
      </span>
    );
  }
  return (
    <button
      type="button"
      className={`value-link${isHighlighted ? " highlight" : ""}`}
      style={{ color }}
      title="Where did this value come from?"
      onClick={() => onValueClick(pointer, value)}
      data-pointer={pointer}
    >
      {text}
    </button>
  );
}

/** Resolve an RFC 6901 pointer against a JSON value. */
export function resolvePointer(doc: unknown, pointer: string): unknown {
  if (pointer === "" || pointer === "/") return doc;
  const tokens = pointer
    .split("/")
    .slice(1)
    .map((t) => t.replace(/~1/g, "/").replace(/~0/g, "~"));
  let current: unknown = doc;
  for (const token of tokens) {
    if (current === null || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[token];
  }
  return current;
}
