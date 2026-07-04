import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { vi } from "vitest";
import Login from "../pages/Login";
import { api } from "../api/client";

// Mock the API
vi.mock("../api/client", () => ({
  api: {
    login: vi.fn(),
  },
}));

const renderWithRouter = (ui) => {
  return render(<BrowserRouter>{ui}</BrowserRouter>);
};

describe("Login Component", () => {
  it("renders login form", () => {
    renderWithRouter(<Login onLogin={vi.fn()} />);

    expect(screen.getByPlaceholderText(/örn: admin/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/••••••••/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Giriş Yap/i })).toBeInTheDocument();
  });

  it("shows loading state when submitting", async () => {
    api.login.mockImplementation(() => new Promise((resolve) => setTimeout(resolve, 100)));

    renderWithRouter(<Login onLogin={vi.fn()} />);

    fireEvent.change(screen.getByPlaceholderText(/örn: admin/i), { target: { value: "admin" } });
    fireEvent.change(screen.getByPlaceholderText(/••••••••/i), { target: { value: "password" } });
    fireEvent.click(screen.getByRole("button", { name: /Giriş Yap/i }));

    expect(screen.getByRole("button")).toHaveTextContent(/Giriş Yapılıyor.../i);
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("shows error message if login failed", async () => {
    api.login.mockRejectedValue(new Error("Giriş başarısız"));

    renderWithRouter(<Login onLogin={vi.fn()} />);

    fireEvent.change(screen.getByPlaceholderText(/örn: admin/i), { target: { value: "admin" } });
    fireEvent.change(screen.getByPlaceholderText(/••••••••/i), { target: { value: "wrongpass" } });
    fireEvent.click(screen.getByRole("button", { name: /Giriş Yap/i }));

    await waitFor(() => {
      expect(
        screen.getByText("Giriş başarısız. Lütfen bilgilerinizi kontrol edin.")
      ).toBeInTheDocument();
    });
  });
});
