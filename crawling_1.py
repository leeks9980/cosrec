from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup

def fetch_full_html(url, output_path="full_page.html"):
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(7) 

        html = driver.page_source

        return html

    finally:
        driver.quit()

def parse_product_info(html):
    soup = BeautifulSoup(html, "html.parser")

    product_info = {}
    for dl in soup.select("#artcInfo dl.detail_info_list"):
        dt = dl.find("dt")
        dd = dl.find("dd")
        if dt and dd:
            title = dt.get_text(strip=True)
            content = dd.get_text(" ", strip=True)
            product_info[title] = content

    return product_info

product_url = "https://www.oliveyoung.co.kr/store/goods/getGoodsDetail.do?goodsNo=A000000170266&dispCatNo=90000010001"
html = fetch_full_html(product_url)
info = parse_product_info(html)

print(" 상품 상세 정보:")
for key, value in info.items():
    print(f"{key} : {value}")


print("\n 저장완료: product_info.txt")
