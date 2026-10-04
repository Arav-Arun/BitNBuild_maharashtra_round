"use client";
import { createContext, useContext } from "react";

export const AppContext = createContext({ staticBundle: false });
export function useAppContext() {
  return useContext(AppContext);
}
