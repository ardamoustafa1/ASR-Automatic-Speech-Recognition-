import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Capture console messages
        page.on("console", lambda msg: print(f"Browser console [{msg.type}]: {msg.text}"))
        page.on("pageerror", lambda err: print(f"Browser error: {err}"))
        
        print("Navigating to http://localhost:8508...")
        await page.goto("http://localhost:8508", wait_until="networkidle")
        
        # Wait a bit for Streamlit to initialize
        await asyncio.sleep(5)
        
        print("Taking screenshot...")
        await page.screenshot(path="screenshot.png", full_page=True)
        
        # Check if the page has Streamlit elements
        content = await page.content()
        if "stApp" in content:
            print("Found Streamlit app container.")
        else:
            print("Could NOT find Streamlit app container!")
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
