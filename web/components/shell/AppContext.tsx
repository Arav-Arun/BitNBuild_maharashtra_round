"use client";
import { createContext, useContext } from "react";

export const AppContext = createContext({ staticBundle: false, mode: null as string | null });
export function useAppContext() {
  return useContext(AppContext);
}
