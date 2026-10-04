"use client";
import { createContext, useContext } from "react";

export interface AppState {
  staticBundle: boolean;
  /** live | recorded | offline, or null while /health has not answered. */
  mode: string | null;
  /** False when /health failed: the API is not running or not reachable. */
  reachable: boolean;
  status: string;
}

export const AppContext = createContext<AppState>({
  staticBundle: false,
  mode: null,
  reachable: true,
  status: "Connecting to recorder",
});
export function useAppContext() {
  return useContext(AppContext);
}
