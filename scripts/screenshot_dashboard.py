"""Screenshot every tab of the running dashboard (used by the
dashboard-preview workflow; needs playwright + a dashboard on :8501)."""

from __future__ import annotations

import pathlib
import unicodedata

from playwright.sync_api import sync_playwright

TABS = ["PnL live", "Positions & risque", "Signaux", "Ordres", "Corrélations", "Santé data", "Méthodo"]
OUT = pathlib.Path("docs/screenshots")
OUT.mkdir(parents=True, exist_ok=True)


def _slug(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return ascii_name.lower().replace(" & ", "_").replace(" ", "_")


def wait_idle(page, timeout_ms: int = 300_000) -> None:
    """Wait until Streamlit's 'running' status widget disappears."""
    page.wait_for_timeout(2000)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="hidden", timeout=timeout_ms)
    except Exception:
        pass
    page.wait_for_timeout(1500)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto("http://localhost:8501", wait_until="domcontentloaded")
        page.wait_for_selector('[data-testid="stTabs"]', timeout=180_000)
        wait_idle(page)
        for i, tab in enumerate(TABS):
            page.get_by_role("tab", name=tab).click()
            wait_idle(page)
            # Streamlit scrolls inside an inner container, so full_page misses
            # anything below the fold: grow the viewport to the content height.
            height = page.evaluate(
                """() => {
                    const sels = ['[data-testid="stMainBlockContainer"]',
                                  '[data-testid="stMain"]', 'section.main',
                                  '[data-testid="stAppViewContainer"]', 'body'];
                    let h = 0;
                    for (const s of sels) {
                        const el = document.querySelector(s);
                        if (el) h = Math.max(h, el.scrollHeight);
                    }
                    return h;
                }"""
            )
            page.set_viewport_size({"width": 1600, "height": min(int(height) + 80, 8000)})
            page.wait_for_timeout(800)
            path = OUT / f"{i}_{_slug(tab)}.png"
            page.screenshot(path=str(path))
            page.set_viewport_size({"width": 1600, "height": 1000})
            print(f"saved {path}")
        browser.close()


if __name__ == "__main__":
    main()
