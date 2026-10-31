import { describe, expect, it, vi } from "vitest";

vi.stubGlobal("matchMedia", () => ({ matches: false }));
const { resolutionFactor } = await import("../src/vr/view3d");

describe("3D quality choice", () => {
  const limit = 350e6;
  it("Full and Half do what they say", () => {
    expect(resolutionFactor("full", 1e9, limit, true)).toBe(1);
    expect(resolutionFactor("half", 1e6, limit, false)).toBe(2);
  });

  it("Auto is full while the volume fits the device", () => {
    expect(resolutionFactor("auto", 157e6, limit, false)).toBe(1);       // CT 300 x 512^2
    expect(resolutionFactor("auto", 314e6, limit, false)).toBe(1);       // CTA 600 x 512^2 on an iPhone
    expect(resolutionFactor("auto", 500e6, limit, false)).toBe(2);
  });

  it("Auto stays at half after memory ran out", () => {
    expect(resolutionFactor("auto", 157e6, limit, true)).toBe(2);
  });
});
