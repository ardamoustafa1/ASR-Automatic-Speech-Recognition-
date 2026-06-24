import { render, screen, waitFor } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { describe, it, expect, vi, beforeEach } from "vitest";
import App from "../App";
import { useAppStore } from "../store/useAppStore";

// Mock matchMedia for recharts ResponsiveContainer
window.matchMedia =
  window.matchMedia ||
  function () {
    return {
      matches: false,
      addListener: function () {},
      removeListener: function () {},
    };
  };

// Mock api client
vi.mock("../api/client", () => ({
  api: {
    alerts: vi.fn().mockResolvedValue([]),
    dashboard: vi.fn().mockResolvedValue({}),
    trends: vi.fn().mockResolvedValue({}),
    topKeywords: vi.fn().mockResolvedValue([]),
  },
}));

describe("App Component", () => {
  beforeEach(() => {
    // Reset the store before each test
    useAppStore.setState({ token: null, alertCount: 0, sidebarOpen: true });
  });

  it("renders login page when there is no token", () => {
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );

    expect(screen.getByText("Kurumsal Giriş")).toBeInTheDocument();
  });

  it("renders the dashboard when token is present", async () => {
    // Set token in Zustand store directly
    useAppStore.setState({ token: "dummy-token" });

    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );

    await waitFor(() => {
      expect(screen.getByText("ASR Command")).toBeInTheDocument();
      expect(screen.getByText("Kelime & Konu Tespiti")).toBeInTheDocument();
    });
  });
});
