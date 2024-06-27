import requests
from bs4 import BeautifulSoup

class Scraper:
    def __init__(self, url, location):
        self.url = url
        self.location = location

    def fetch_content(self):
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            print(f"Error fetching {self.url}: {e}")
            return None

    def parse_content(self, content):
        try:
            soup = BeautifulSoup(content, 'html.parser')
            return soup.select(self.location)
        except Exception as e:
            print(f"Error parsing content: {e}")
            return None

    def get_scraped_data(self):
        content = self.fetch_content()
        if content:
            parsed_data = self.parse_content(content)
            return parsed_data
        return None