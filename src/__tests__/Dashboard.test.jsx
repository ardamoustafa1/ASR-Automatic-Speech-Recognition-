import { render, screen, waitFor } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { vi } from "vitest";
import Dashboard from "../pages/Dashboard";

// Mock the API
vi.mock("../api/client", () => ({
  api: {
    dashboard: vi.fn().mockResolvedValue({
      hits_today: 10,
      conversations_total: 5,
      active_alerts: 2,
      top_rising_keyword: { keyword: "iptal", increase: 50 },
    }),
    topKeywords: vi.fn().mockResolvedValue([]),
    trends: vi.fn().mockResolvedValue({
      keyword: "iptal",
      current_count: 5,
      current_conversations: 3,
      pct_change: 50,
    }),
  },
}));

const renderWithRouter = (ui) => {
  return render(<BrowserRouter>{ui}</BrowserRouter>);
};

describe("Dashboard Component", () => {
  it("renders dashboard headers", async () => {
    renderWithRouter(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText(/Bugünkü Eşleşmeler/i)).toBeInTheDocument();
      expect(screen.getByText(/Toplam Görüşme/i)).toBeInTheDocument();
      expect(screen.getByText(/Aktif Uyarılar/i)).toBeInTheDocument();
      expect(screen.getByText(/En Çok Artan/i)).toBeInTheDocument();
    });
  });
});
