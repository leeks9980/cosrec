from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup
import pyautogui
import csv
import pandas as pd
import mango.prd_info.Crawling_Product_Information_Preparation as CPI
import os

os.chdir(r'C:\Users\leeks\my_git\mango\prd_info\url')  #Parts that need to be fixed/ Location of url file

files = os.listdir('url')

prd_dict = {}
for i in files:
    Temporary_list = []
    read_csv = pd.read_csv(i)
    for i in range(len(read_csv)):
        Temporary_list.append(read_csv.iloc[i,1])
        prd = read_csv.iloc[i,0]
    prd_dict[prd] =Temporary_list
print(prd_dict.keys())
My_part = input('원하는 항목 선택')

Category_List = My_part.split('  ')
Category_dict = {i:prd_dict[i] for i in Category_List}

b = []
df = pd.DataFrame(columns=['제품명', '가격', '사용기한', '사용방법', '큰카테고리', '작은카테고리'])

for k,v in Category_dict.items(): 
    for url in v:
        html = CPI.fetch_full_html(url)
        info, ingredient = CPI.parse_product_info(html, k)
        b.append(ingredient)
        df.loc[len(df)] = info

df2 = pd.DataFrame(b)
df.to_csv('C:/Users/leeks/OneDrive/바탕 화면/csv/Product_Information.csv', index=False)  #Parts that need to be fixed/ Save Path
df2.to_csv('C:/Users/leeks/OneDrive/바탕 화면/csv/Ingredients_list.csv', index=False)  #Parts that need to be fixed/ Save Path
