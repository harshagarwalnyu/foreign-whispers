"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Returns a live elapsed-ms counter that ticks every second while
 * `startedAt` is truthy. Returns `undefined` when inactive.
 */
export function useElapsed(startedAt: number | undefined): number | undefined {
  const [elapsed, setElapsed] = useState<number | undefined>(undefined);
  const startedAtRef = useRef(startedAt);
  startedAtRef.current = startedAt;

  useEffect(() => {
    if (!startedAt) {
      setElapsed(undefined);
      return;
    }
    const tick = () => {
      if (startedAtRef.current) setElapsed(Date.now() - startedAtRef.current);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [startedAt]);

  return elapsed;
}
