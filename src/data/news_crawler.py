"""
News crawler module for collecting financial news articles.
Adapted from the YNA crawler to ensure compatibility with existing data format.
"""

import os
import csv
import time
import logging
import datetime
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse, urlunparse
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsCrawler:
    """
    News crawler for collecting financial news articles from YNA (Yonhap News Agency).
    Ensures compatibility with the existing data format.
    """
    
    def __init__(self, output_dir: str = None, use_ai_summary: bool = True, headless: bool = True):
        """
        Initialize the news crawler.
        
        Args:
            output_dir: Directory to save crawled news data
            use_ai_summary: Whether to extract AI summary when available
            headless: Whether to run browser in headless mode
        """
        self.output_dir = output_dir or os.path.join(os.getcwd(), "data", "news")
        self.use_ai_summary = use_ai_summary
        self.headless = headless
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized NewsCrawler with output_dir={self.output_dir}, use_ai_summary={use_ai_summary}")
    
    def init_driver(self) -> webdriver.Chrome:
        """
        Initialize the Chrome WebDriver.
        
        Returns:
            Configured Chrome WebDriver instance
        """
        options = Options()
        if self.headless:
            options.add_argument('--headless')
        
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--start-maximized')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # Use webdriver_manager to handle driver installation
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        return driver
    
    def clean_article_url(self, url: str) -> str:
        """
        Clean article URL by removing query parameters.
        
        Args:
            url: Original article URL
            
        Returns:
            Cleaned URL
        """
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
    
    def scrape_search_results(self, date: str, keyword: str) -> List[Dict[str, str]]:
        """
        Scrape search results for a specific date and keyword.
        
        Args:
            date: Date in YYYY-MM-DD format
            keyword: Search keyword (e.g., company name or stock code)
            
        Returns:
            List of dictionaries with title and link for each article
        """
        base_url = f"https://www.yna.co.kr/search/index?query={keyword}&period=sel&from={date}&to={date}&ctype=A"
        driver = self.init_driver()
        articles = []
        page_no = 1
        
        try:
            logger.info(f"Scraping search results for {keyword} on {date}")
            
            while True:
                url = base_url if page_no == 1 else f"{base_url}&page_no={page_no}"
                driver.get(url)
                time.sleep(5)  # Wait for page to load
                
                results = driver.find_elements(By.XPATH, "//ul[@class='list01']//li//a[contains(@href, '/view/')]")
                if not results:
                    logger.info(f"No more results found on page {page_no}")
                    break
                
                for result in results:
                    link = result.get_attribute("href")
                    title = result.text.strip()
                    if link and title:
                        articles.append({"title": title, "link": link})
                
                logger.info(f"Found {len(results)} articles on page {page_no}")
                
                if len(results) < 10:  # Assuming 10 results per page
                    break
                    
                page_no += 1
        except Exception as e:
            logger.error(f"Error scraping search results: {str(e)}")
        finally:
            driver.quit()
        
        logger.info(f"Total articles found: {len(articles)}")
        return articles
    
    def scrape_article_details(self, url: str) -> Dict[str, Any]:
        """
        Scrape details of a specific article.
        
        Args:
            url: Article URL
            
        Returns:
            Dictionary with article details
        """
        driver = self.init_driver()
        
        try:
            logger.info(f"Scraping article details from {url}")
            driver.get(url)
            time.sleep(5)  # Wait for page to load
            
            # Title extraction
            title = driver.find_element(By.TAG_NAME, "h1").text.strip()
            
            # Date extraction
            date_element = driver.find_element(By.CSS_SELECTOR, "div.update-time")
            date_str = date_element.get_attribute("data-published-time")
            # Convert from YYYY-MM-DD HH:MM to YYYYMMDD HH:MM format
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M").strftime("%Y%m%d %H:%M")
            
            # Body extraction
            body = driver.find_element(By.CSS_SELECTOR, "div.article").text.strip()
            
            # Emotion extraction
            emotions = driver.find_elements(By.CSS_SELECTOR, "div.empathy-zone button")
            emotion_data = {
                e.get_attribute("data-emotion"): e.text.strip()
                .replace("좋아요", "")
                .replace("슬퍼요", "")
                .replace("화나요", "")
                .replace("후속요청", "")
                .strip()
                for e in emotions
            }
            
            # Num_comment extraction
            try:
                num_comment = driver.find_element(By.CSS_SELECTOR, "button.btn-type422.cmt01 span.color21").text.strip()
            except NoSuchElementException:
                num_comment = "0"
            
            # AI Summary extraction via JavaScript (if enabled)
            ai_summary = ""
            if self.use_ai_summary:
                ai_summary = driver.execute_script("""
                    let summaryElement = document.querySelector("article.story-summary");
                    return summaryElement ? summaryElement.innerText.trim() : "";
                """)
            
            # Create article data dictionary
            article_data = {
                "Title": title,
                "Date": date,
                "Press": "yna",
                "Link": self.clean_article_url(url),
                "Body": body,
                "Emotion": emotion_data,
                "Num_comment": num_comment,
                "AI Summary": ai_summary
            }
            
            logger.info(f"Successfully scraped article: {title}")
            return article_data
            
        except Exception as e:
            logger.error(f"Error scraping article details: {str(e)}")
            return {}
        finally:
            driver.quit()
    
    def save_to_csv(self, data: List[Dict[str, Any]], filename: str) -> None:
        """
        Save scraped data to CSV file.
        
        Args:
            data: List of article data dictionaries
            filename: Output filename
        """
        if not data:
            logger.warning("No data to save")
            return
            
        filepath = os.path.join(self.output_dir, filename)
        cols = ["Title", "Date", "Press", "Link", "Body", "Emotion", "Num_comment", "AI Summary"]
        
        try:
            with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
                    
            logger.info(f"Saved {len(data)} articles to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data to CSV: {str(e)}")
    
    def crawl_by_date(self, date: str, keyword: str) -> int:
        """
        Crawl articles for a specific date and keyword.
        
        Args:
            date: Date in YYYY-MM-DD format
            keyword: Search keyword
            
        Returns:
            Number of articles crawled
        """
        logger.info(f"Crawling articles for {keyword} on {date}")
        
        # Get search results
        search_results = self.scrape_search_results(date, keyword)
        
        if not search_results:
            logger.warning(f"No articles found for {keyword} on {date}")
            return 0
        
        # Scrape article details
        scraped_data = []
        for article in tqdm(search_results, desc=f"Scraping articles for {date}", leave=False):
            details = self.scrape_article_details(article['link'])
            if details:
                scraped_data.append(details)
        
        # Save to CSV
        date_for_filename = date.replace("-", "")
        filename = f"yna_{keyword}_{date_for_filename}.csv"
        self.save_to_csv(scraped_data, filename)
        
        return len(scraped_data)
    
    def crawl_date_range(self, start_date: str, end_date: str, keyword: str) -> int:
        """
        Crawl articles for a date range and keyword.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            keyword: Search keyword
            
        Returns:
            Total number of articles crawled
        """
        logger.info(f"Crawling articles for {keyword} from {start_date} to {end_date}")
        
        # Parse dates
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end - start).days + 1
        
        # Generate date list
        dates = [(start + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(total_days)]
        
        # Crawl articles for each date
        total_articles = 0
        for date in tqdm(dates, desc="Overall crawling progress"):
            articles_count = self.crawl_by_date(date, keyword)
            total_articles += articles_count
        
        logger.info(f"Total articles crawled: {total_articles}")
        return total_articles
    
    def crawl_for_stock(self, stock_code: str, year: Union[str, int] = None) -> int:
        """
        Crawl articles for a specific stock code and year.
        
        Args:
            stock_code: Stock code (e.g., "005930" for Samsung)
            year: Year to crawl (defaults to current year if not specified)
            
        Returns:
            Number of articles crawled
        """
        # Set year to current year if not specified
        if year is None:
            year = datetime.datetime.now().year
            
        year_str = str(year)
        
        # Set date range for the year
        start_date = f"{year_str}-01-01"
        end_date = f"{year_str}-12-31"
        
        # Crawl articles
        return self.crawl_date_range(start_date, end_date, stock_code)


# Example usage
if __name__ == "__main__":
    # Create crawler instance
    crawler = NewsCrawler()
    
    # Crawl articles for Samsung (005930) for 2025
    articles_count = crawler.crawl_for_stock("005930", 2025)
    print(f"Total articles crawled: {articles_count}")
