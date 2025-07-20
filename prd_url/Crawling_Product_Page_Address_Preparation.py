from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup
import random

#Page parsing
def fetch_full_html(url, output_path="full_page.html"):
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(random.uniform(10, 20)) 
        html = driver.page_source
        return html

    finally:
        driver.quit()

# find product information from the HTML
def Product_Information_Address(html):
    soup = BeautifulSoup(html, "html.parser")
    url = []
    for info_div in soup.find_all("div", class_="prd_info"):
        a_tag = info_div.find("a", href=True)
        if a_tag:
            url.append(a_tag["href"])
    return url
