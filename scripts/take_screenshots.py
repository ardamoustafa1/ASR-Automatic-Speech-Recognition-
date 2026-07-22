import asyncio
import os
import shutil
import subprocess

from playwright.async_api import async_playwright


async def main():
    os.makedirs("docs/screenshots", exist_ok=True)
    os.makedirs("docs/assets", exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        # 1. Take React Screenshots
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        print("Taking Dashboard screenshot...")
        await page.goto("http://localhost:5173/")
        await page.wait_for_timeout(2000)
        await page.screenshot(path="docs/screenshots/dashboard_dark.png")

        print("Taking Analytics screenshot...")
        await page.goto("http://localhost:5173/analytics")
        await page.wait_for_timeout(2000)
        await page.screenshot(path="docs/screenshots/analytics_trend.png")

        # Record a short video for Live ASR
        print("Recording Live ASR demo...")
        video_context = await browser.new_context(
            record_video_dir="docs/assets/", record_video_size={"width": 1280, "height": 800}
        )
        video_page = await video_context.new_page()
        await video_page.goto("http://localhost:5173/live")
        await video_page.wait_for_timeout(2000)
        # Type something to show interaction
        await video_page.evaluate(
            "document.body.innerHTML += '<div style=\"position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: var(--accent); padding: 20px; border-radius: 12px; font-size: 24px; color: white; z-index: 9999;\">🎙️ Canlı Dinleme Aktif...<br><br><small>Test Ses Verisi İşleniyor...</small></div>'"
        )
        await video_page.wait_for_timeout(3000)
        await video_context.close()  # Saves the video

        # Rename the random webm file to demo.webm
        for file in os.listdir("docs/assets"):
            if file.endswith(".webm"):
                os.rename(f"docs/assets/{file}", "docs/assets/demo.webm")
                break

        # Use ffmpeg to convert to gif
        print("Converting WebM to GIF...")
        subprocess.run(
            [
                shutil.which("ffmpeg") or "ffmpeg",
                "-y",
                "-i",
                "docs/assets/demo.webm",
                "-vf",
                "fps=10,scale=800:-1:flags=lanczos",
                "docs/assets/demo.gif",
            ],
            capture_output=True,
        )

        # 2. Take Streamlit Screenshot
        print("Taking Streamlit screenshot...")
        context_st = await browser.new_context(viewport={"width": 1280, "height": 800})
        page_st = await context_st.new_page()
        await page_st.goto("http://localhost:8501")
        await page_st.wait_for_timeout(3000)
        await page_st.screenshot(path="docs/screenshots/streamlit_8501.png")

        # 3. Generate Architecture Diagram using Mermaid via Playwright
        print("Generating Architecture diagram...")
        mermaid_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({ startOnLoad: true, theme: 'dark' });
            </script>
            <style>body { background: #0d0d12; color: white; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }</style>
        </head>
        <body>
            <div class="mermaid">
graph TD
    Agent((Call Center Agent))
    Supervisor((Supervisor))
    Client["React Dashboard<br/>Zustand · Vite · Recharts"]
    API["FastAPI Gateway<br/>uvicorn · WebSockets · SlowAPI"]
    Auth["Auth Service<br/>JWT · Bcrypt · RBAC"]
    ASREngine["ASR Service (Singleton)<br/>faster-whisper · mlx-whisper"]
    Sentiment["Sentiment Engine<br/>mDeBERTa Zero-Shot"]
    Churn["Churn Engine<br/>WPM + Temporal Weighting"]
    Empathy["Empathy Engine<br/>Fuzzy Match + AI Gate"]
    Compliance["Compliance Engine<br/>Negation Filter + NLP"]
    Keywords["Keyword Engine<br/>Exact · Fuzzy · Semantic · Regex"]
    Trend["Trend Engine<br/>Z-Score · Linear Regression"]
    DB[(PostgreSQL / SQLite<br/>SQLAlchemy ORM)]
    Cache[(Redis / In-Memory<br/>FastAPI-Cache2)]
    Prometheus[(Prometheus Metrics<br/>/metrics endpoint)]
    Agent --> Client
    Supervisor --> Client
    Client --> API
    API --> Auth
    API --> ASREngine
    ASREngine --> Sentiment
    ASREngine --> Churn
    ASREngine --> Empathy
    ASREngine --> Compliance
    ASREngine --> Keywords
    Keywords --> Trend
    API --> DB
    API --> Cache
    API -.-> Prometheus
    classDef frontend fill:#3b82f6,stroke:#1d4ed8,color:white;
    classDef backend fill:#10b981,stroke:#047857,color:white;
    classDef ai fill:#8b5cf6,stroke:#6d28d9,color:white;
    classDef database fill:#f59e0b,stroke:#b45309,color:white;
    class Client frontend;
    class API,Auth backend;
    class ASREngine,Sentiment,Churn,Empathy,Compliance,Keywords,Trend ai;
    class DB,Cache,Prometheus database;
            </div>
        </body>
        </html>
        """
        with open("mermaid.html", "w") as f:
            f.write(mermaid_html)

        page_arch = await context.new_page()
        await page_arch.goto(f"file://{os.path.abspath('mermaid.html')}")
        await page_arch.wait_for_timeout(3000)

        # Take screenshot of the diagram element specifically to crop correctly
        element_handle = await page_arch.query_selector(".mermaid")
        if element_handle:
            await element_handle.screenshot(path="docs/assets/architecture.png")
        else:
            await page_arch.screenshot(path="docs/assets/architecture.png")

        os.remove("mermaid.html")

        await browser.close()
        print("All visual assets generated successfully!")


if __name__ == "__main__":
    asyncio.run(main())
