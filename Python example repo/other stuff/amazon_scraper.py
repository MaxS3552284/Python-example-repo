# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 11:22:21 2021

@author: Max
"""
# AMAZON SCRAPER 
# tutorial: https://www.youtube.com/watch?v=_AeudsbKYG8


import csv
from bs4 import BeautifulSoup
import requests
import pandas as pd
from time import sleep
import os.path
import time # for timer

# for firefoy or chrome
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from datetime import date
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


# fro microsoft edge
# from msedge.selenium_tools import Edge, EdgeOtions

# STARTUP INSTANCE OF WEBDRIVER:
# for ff or chrome
# driver = webdriver.FireFox()

# options = Options()
# options.binary_location = "C:\Program Files\Google\Chrome\Application\chrome.exe" 
# # specify in otions where your chrome.exe is, if its not the default location
# driver = webdriver.Chrome(chrome_options = options, executable_path=r'C:\Webdrivers\chromedriver.exe')
# # specify that modded options and path to chromedriver.exe shall be used

# # for edge
# # options = EdgeOtions()
# # options.use_chromium = True
# # driver = Edge(options=options)

# #

# # url = 'https://www.amazon.com'
# url_base = 'https://www.amazon.de'
# driver.get(url)

# TIMER FOR REPLACEMENT OF TICTOC IN MATLAB:
    
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    
    
TicToc2 = TicTocGenerator() # create another instance of the TicTocGen generator

def toc2(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc2
    tempTimeInterval = next(TicToc2)
    if tempBool:
        print( "Elapsed time 2: %f seconds.\n" %tempTimeInterval )

def tic2():
    # Records a time in TicToc2, marks the beginning of a time interval
    toc2(False)

def get_url(search_term):
    # generate an url for product search
    template = 'https://www.amazon.de/s?k={}&ref=nb_sb_ss_ts-doa-p_1_3'
    search_term = search_term.replace(' ','+') 
    # replace spacebars from searchterm with +'s in url
    # return modified link, {} in template will be replaced with search term
    
    # add term query to url (for changeing side numbers)
    url = template.format(search_term)
    # add page query placeholder, count pagenumber up
    url += '&page={}'
    
    
    return url

def largestNumber(number_string):
    la=[int(x) for x in number_string.split() if x.isdigit()]
    # split string on every ' '(spacebar seemingly=defaut), to get a list
    # for every part of list, check if digit
    # if digit, convert to int and put into list
    
    # return max. value in list
    return max(la) if la else None

# test
# url = get_url('ultrawide monitor')
# # print(url)
# driver.get(url)

# # EXTRACT CONTENT OF PAGE

# # create soup object and get html content
# soup = BeautifulSoup(driver.page_source, 'html.parser')
# # %%
# # use subject to extract all div-tag data_component_type with value of s-search-result (the single products on amazon page) 
# results = soup.find_all('div', {'data-component-type': 's-search-result'})
# number_results = len(results) # 16 search results + 6 sponsored content

# # prototype the extraction of a single record
# item = results[0]
# atag = item.h2.a
# item_name = atag.text
# print(item_name)
# description = atag.text.strip() #remove extra empty text at start and end of name
# print(description)
# hyperlink_part = atag.get('href')
# full_hyperlink = url_base + atag.get('href')
# driver.get(full_hyperlink)

# remember.de, .com etc. will cause differences in link and infalidate it
# https://www.amazon.com/gp/slredirect/picassoRedirect.html/ref=pa_sp_atf_aps_sr_pg1_1?ie=UTF8&adId=A0482780VZZ1Q69E7Z6S&url=%2FLG-34WN80C-B-IPS-H%25C3%25B6henverstellbar-Multitasking%2Fdp%2FB083QT6Z8R%2Fref%3Dsr_1_1_sspa%3Fdchild%3D1%26keywords%3Dultrawide%2Bmonitor%26qid%3D1630752535%26sr%3D8-1-spons%26psc%3D1&qualifier=1630752535&id=6498279014393148&widgetName=sp_atf
# https://www.amazon.de/gp/slredirect/picassoRedirect.html/ref=pa_sp_atf_aps_sr_pg1_1?ie=UTF8&adId=A0482780VZZ1Q69E7Z6S&url=%2FLG-34WN80C-B-IPS-H%25C3%25B6henverstellbar-Multitasking%2Fdp%2FB083QT6Z8R%2Fref%3Dsr_1_1_sspa%3Fdchild%3D1%26keywords%3Dultrawide%2Bmonitor%26qid%3D1630754182%26sr%3D8-1-spons%26psc%3D1&qualifier=1630754182&id=6544718027054998&widgetName=sp_atf

# locate price and get it
# price_parent = item.find('span', 'a-price')
# price = price_parent.find('span', 'a-offscreen').text
# #without .text, price = <span class="a-offscreen">618,83 €</span>
# print(price)
# rating = item.i.text # rating of item
# # review_counter = item.find('span', {'class': 'a-size-base', 'dir': 'auto'}).text # {} = dictionary
# review_counter = item.find('span', {'class': 'a-size-base'}).text


# # %%
def extract_record(item):
    # extract and return infos from a single record
    
    # description and url
    url_base = 'https://www.amazon.de'
    atag = item.h2.a
    description = atag.text.strip() #remove extra empty text at start and end of name
    full_hyperlink = url_base + atag.get('href')
    
    try: # error handling in case of empty values
        # price
        price_parent = item.find('span', 'a-price')
        price = price_parent.find('span', 'a-offscreen').text
    except AttributeError:
        return
        
    try:
        rating = item.i.text # rating of item
        # review_counter = item.find('span', {'class': 'a-size-base', 'dir': 'auto'}).text # {} = dictionary
        review_counter = item.find('span', {'class': 'a-size-base'}).text
    except AttributeError:
        rating = ''
        review_counter = ''

    # save info in tuple
    result = (description, price, rating, review_counter, full_hyperlink)
    
    return(result)

def main(search_term, page_number = 20):

    
    options = Options()
    options.binary_location = "C:\Program Files\Google\Chrome\Application\chrome.exe" 
    # specify in otions where your chrome.exe is, if its not the default location
    driver = webdriver.Chrome(chrome_options = options, executable_path=r'C:\Webdrivers\chromedriver.exe')
    # specify that modded options and path to chromedriver.exe shall be used
    
    TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
    TicToc2 = TicTocGenerator() # create an instance of the TicTocGen generator
        
    
    if os.path.isfile('Amazon_Scraper_Result '+ search_term +'.csv'): # #ok %% ~ = logical NOT argument

        print("Loading "+'Amazon_Scraper_Result '+ search_term +'.csv'+" ...")
        result2 = pd.read_csv('Amazon_Scraper_Result '+ search_term +'.csv', sep=';')
        print("Finished loading.")

    else:
    

        # for edge
        # options = EdgeOtions()
        # options.use_chromium = True
        # driver = Edge(options=options)
    
        # url = 'https://www.amazon.com'
        # url_base = 'https://www.amazon.de'
    
        records = []
        url = get_url(search_term)
        
        for page in range(1,(page_number+1)): # not inclusive of last number, and pages start with 1, not 0
            
            driver.get(url.format(page))
            # EXTRACT CONTENT OF PAGE
            # create soup object and get html content
            soup = BeautifulSoup(driver.page_source, 'html.parser')  
    
            results = soup.find_all('div', {'data-component-type': 's-search-result'})
        
            for item in results:
                record = extract_record(item)
                
                if record: # is not empty
                    records.append(record)
                    
        # driver.close() # close driver after every page
        
        # for row in records:
        #     print (row[1])
        
        # save data to .csv file
        # with open('results.csv', 'w', newline = '', encoding = 'utf-8') as f:
        #     writer = csv.writer(f, delimiter = ';')
        #     writer.writerow(['Description', 'Price', 'Rating', 'Review_Counter', 'full_Hyperlink', 'Quantity in Stock', 'Quantity'])
        #     for row in records:
        #         writer.writerow(row)
        #     # writer.writerows(records)
      
        # ALTERNATIV:
        # page_number =20
        result2_columns = ['Description', 'Price', 'Rating', 'Review_Counter', 'full_Hyperlink', 'Quantity in Stock', 'Quantity'];
        result2_index = range(page_number);
        result2 = pd.DataFrame(index=result2_index, columns=result2_columns);
        
        for d in range(len(records)):
            records_tuple = records[d]
            
            for i in range(len(records_tuple)):
                
                result2.loc[d, result2_columns[i]] = records_tuple[i]
    
    today = date.today()
    current_date= 'Quantity: ' + today.strftime("%d/%m/%Y")
    
    tic()
    
    for l in range(len(result2)):
        print("Amazon Scraper is now fetching the stock quantity of item " + str(l+1) + " of " + str(len(result2)) + " for today, the " + today.strftime("%d/%m/%Y") + ". ")
        tic2()
        hyper_url = result2['full_Hyperlink'][l]
        driver.get(hyper_url)
        timeout = 10
        try:
            element_present = EC.presence_of_element_located((By.ID, 'quantity'))
            WebDriverWait(driver, timeout).until(element_present)
        except TimeoutException:
            print ("Timed out waiting for element to load. Wait additional 3 Sec. If you still get an error after this time, your i-net connection probably wiped or your browser hung itself. Or Product is out off stock.")
            # sleep(3)
        try:
            quant_drop_down = driver.find_element_by_id("quantity").text
            quant = largestNumber(quant_drop_down)
        except NoSuchElementException:
            quant = 1
                
        result2.loc[l, (current_date)] = quant
        toc2()
        
    toc()
    driver.close() # close driver after every page
    
    result2.to_csv('Amazon_Scraper_Result '+ search_term +'.csv', sep = ';', index = False)




# main('ultrawide monitor', page_number = 10)
# b1=0


search_term_list =['beleuchtung garten metall stahl', 'lightbox metall stahl', 'gartenstecker halloween metall', 'gartenstecker weihnachten christmas metall', 'kaminholzregal metall innen außen', 'mülleimer abfalleimer müllbox abfallbox metall', 'paketbox briefbox metall', 'wandbild metall edelstahl']
# page_number = 10
for i in range(len(search_term_list)):
    
    search_term = search_term_list[i]
    main(search_term, page_number=10)



























