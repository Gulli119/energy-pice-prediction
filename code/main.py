from scraper import Scraper

# Example usage:
if __name__ == "__main__":
    url = "http://example.com"
    location = "h1"  # CSS selector for the elements you want to scrape

    scraper = Scraper(url, location)
    data = scraper.get_scraped_data()

    if data:
        for element in data:
            print(element.get_text())
    else:
        print("No data found.")