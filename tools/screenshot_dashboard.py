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
        if pg.locator('[data-testid="stSpinner"]').count() == 0:
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
    settle(pg, "How the ensemble families work")
    shot(pg, "01_home")
    go(pg, "Ensemble Arena", "Bagging effect")
    shot(pg, "02_arena")
    go(pg, "RF vs XGBoost", "RF mean CV RMSE")
    shot(pg, "03_rf_xgb")
    shot(pg, "04_rf_xgb_cv", scroll=520)
    go(pg, "Hybrid ARIMA + RF", "Hybrid RMSE")
    shot(pg, "05_hybrid")
    shot(pg, "06_hybrid_seasons", "Which model wins in which monsoon season")
    go(pg, "Hybrid RF + ANN", "error correlation")
    shot(pg, "07_rf_ann")
    go(pg, "Tomorrow in Colombo", "Heat alert")
    shot(pg, "08_tomorrow")
    shot(pg, "09_tomorrow_chart", scroll=560)
    try:                                   # what-if: a very rainy, cloudy last day (rain slider to max, radiation to min)
        pg.mouse.move(1100, 500)
        pg.mouse.wheel(0, -3000)
        time.sleep(2)
        thumbs = pg.locator('[data-testid="stSlider"] [role="slider"]')
        thumbs.nth(0).focus(); thumbs.nth(0).press("End"); time.sleep(4)
        settle(pg, "Heat alert")
        thumbs.nth(1).focus(); thumbs.nth(1).press("Home"); time.sleep(4)
        settle(pg, "from your what-if", timeout=120_000)
        shot(pg, "08b_tomorrow_whatif")
    except Exception as e:
        print("what-if screenshot skipped:", e, flush=True)
    go(pg, "Research & About", "Random Forest vs gradient boosting")
    shot(pg, "10_about")
    browser.close()

(OUT / "pages.txt").write_text("\n".join(LOG), encoding="utf-8")
print("done")
