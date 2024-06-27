import requests


def check_usa_war(api_key):
    query = 'RUSSIA AND (war OR conflict OR military action)'
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&language=en&sortBy=publishedAt'
    response = requests.get(url)
    articles = response.json().get('articles', [])

    # Check for articles indicating USA entering a war
    for article in articles:
        title = article['title'].lower()
        description = article['description'].lower()
        if 'russia' in title or 'russia' in description:
            if 'war' in title or 'conflict' in title or 'military action' in title or \
                    'war' in description or 'conflict' in description or 'military action' in description:
                return 1  # War detected
    return 0  # No war detected


api_key = 'insert your key'
war_status = check_usa_war(api_key)
print(war_status)