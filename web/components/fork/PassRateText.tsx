import type { PassRate } from "../../lib/contract";

/** "4/5 passed (95% CI 38–96%)" */
export function PassRateText({ rate }: { rate: PassRate }) {
  return (
    <span title="Wilson 95% interval">
      <strong>{rate.passed}/{rate.total}</strong> passed
      <span className="faint"> (95% CI {Math.round(rate.ci_low * 100)}–{Math.round(rate.ci_high * 100)}%)</span>
    </span>
  );
}
