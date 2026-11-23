"""Take full-HD screenshots of every dashboard page (used in the project video).
Needs the app running at http://localhost:8501 - see .github/workflows/screenshots.yml."""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("docs/screenshots")
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8501"
LOG = []


def settle(pg, text, timeout=900_000):
    """Wait until `text` is on the page and Streamlit has finished running."""
    pg.get_by_text(text).first.wait_for(timeout=timeout)
    t0 = time.time()
    while time.time() - t0 < 600:
        busy = pg.locator('[data-testid="stSpinner"]').count()
        if busy == 0:
            break
        time.sleep(1)
    time.sleep(4)


def go(pg, label, wait_text):
    pg.locator('[data-testid="stSidebar"] label').filter(has_text=label).first.click()
    settle(pg, wait_text)


def shot(pg, name, anchor=None, scroll=0):
    if anchor:
        pg.get_by_text(anchor).first.scroll_into_view_if_needed()
        time.sleep(2)
    if scroll:
        pg.mouse.move(1100, 500)
        pg.mouse.wheel(0, scroll)
        time.sleep(2)
    pg.screenshot(path=str(OUT / f"{name}.png"))
    body = pg.locator("body").inner_text()
    bad = any(k in body for k in ("Traceback", "Exception", "Error:"))
    LOG.append(f"===== {name} | problem={bad}\n{body[:2500]}\n")
    print(f"saved {name}.png  problem={bad}", flush=True)


with sync_playwright() as p:
    browser = p.chromium.launch()
    pg = browser.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1.2)
    pg.goto(URL)
    settle(pg, "How the three ensemble families work")
    shot(pg, "01_home")

    go(pg, "Ensemble Arena", "Best bagging AUC")
    shot(pg, "02_arena")
    shot(pg, "03_arena_trees", "how does accuracy grow")
    shot(pg, "04_arena_threshold", "Decision threshold explorer")

    go(pg, "Churn Predictor", "Stacking ensemble churn probability")
    shot(pg, "05_predictor")
    pg.get_by_role("tab").filter(has_text="Retention budget optimizer").first.click()
    time.sleep(3)
    shot(pg, "06_optimizer", "Optimizer net ROI")

    go(pg, "Deal Checker", "Hybrid RF+ANN estimate")
    shot(pg, "07_deal")
    shot(pg, "08_deal_hist", scroll=650)

    go(pg, "Colombo Weather", "Hybrid test RMSE")
    shot(pg, "09_weather")
    shot(pg, "10_weather_detail", "How the hybrid works")

    go(pg, "Research & About", "Random Forest vs gradient boosting")
    shot(pg, "11_about")
    browser.close()

(OUT / "pages.txt").write_text("\n".join(LOG), encoding="utf-8")
print("done")
