# scripts/backfill_newsapi.py
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

newsapi = NewsApiClient(api_key='52628d63335d4c7ea0e6604fe97ef947')

def backfill_with_multiple_queries():
    """Get articles using multiple targeted queries to maximize coverage."""
    
    # Define specific topic queries
    queries = [
        'federal reserve OR fed OR interest rates',
        'stock market OR dow jones OR nasdaq OR s&p 500',
        'artificial intelligence OR AI OR machine learning',
        'climate change OR global warming OR COP28',
        'election OR trump OR biden OR politics',
        'ukraine OR russia OR war',
        'israel OR gaza OR palestine OR hamas',
        'inflation OR recession OR economy',
        'covid OR pandemic OR health',
        'technology OR silicon valley OR tech industry',
        'ai OR chatgpt OR openai OR google ai',
        'bitcoin OR crypto OR blockchain OR ethereum',
        'trump past quotes or trump speech quotes',
        'Zohran Mamndani NYC mayor victory',
        'Immigration customs enforcement and deportation',
        'GOP shutdown of the government',
        'Minnesota ICE shootings',
        'Texas School Shooting',
        'AI and China',
        'AI Bubble and investments outweighing demand',
        'Intel and Qualcomm',
        'Apple, Nvidia, and Meta',
        'GPU and AI Chips/Hardware',
        'Oura Sleep Tracking Ring',
        'The New York Times and The Wall Street Journal',
        'TikTok and the Chinese government',
        'Iphone17',
        'Rising cost of living and infflation',
        'mass layoffs and job cuts',
        'Federal spending and budget cuts',
        'Black Holes',
        'NASA',
        'Increased unemployment and jobblessness',
        'Investments and unprofitability',
        'Trumps trade policy',
        'Federal Reserve Bank'
    ]
    
    all_articles = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    total_articles_fetched = 0
    MAX_ARTICLES = 100  # Free tier limit
    
    for query in queries:
        if total_articles_fetched >= MAX_ARTICLES:
            print(f"\n⚠️  Reached 100 article limit. Stopping.")
            break
            
        print(f"\nFetching articles for: {query}")
        
        try:
            # Calculate how many articles we can still fetch
            remaining = MAX_ARTICLES - total_articles_fetched
            page_size = min(100, remaining)  # Don't exceed limit
            
            response = newsapi.get_everything(
                q=query,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt',
                page_size=page_size,
                page=1  # Only fetch first page
            )
            
            total_results = response['totalResults']
            print(f"  Total results available: {total_results}")
            print(f"  Fetching: {page_size} articles")
            
            articles_added = 0
            for article in response['articles']:
                # Deduplicate by URL
                if article['url'] not in [a['article_id'] for a in all_articles]:
                    all_articles.append({
                        'article_id': article['url'],
                        'title': article['title'],
                        'url': article['url'],
                        'published': article['publishedAt'],
                        'source': article['source']['name'],
                        'summary': article['description'],
                        'content': article.get('content', ''),
                        'query_matched': query,
                    })
                    articles_added += 1
                    total_articles_fetched += 1
                    
                    if total_articles_fetched >= MAX_ARTICLES:
                        break
            
            print(f"  ✓ Added {articles_added} unique articles (Total: {len(all_articles)})")
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"  ✗ Error with query '{query}': {e}")
            continue
    
    # Save
    output_dir = Path('/Users/vipulpandey/Desktop/eval_system//articles')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'newsapi_backfill.jsonl'
    
    with open(output_file, 'w') as f:
        for article in all_articles:
            f.write(json.dumps(article) + '\n')
    
    print(f"\n{'='*60}")
    print(f"✓ Backfill complete!")
    print(f"  Total unique articles: {len(all_articles)}")
    print(f"  Total articles fetched: {total_articles_fetched}/{MAX_ARTICLES}")
    print(f"  Saved to: {output_file}")
    print(f"{'='*60}")
    
    return all_articles

if __name__ == '__main__':
    backfill_with_multiple_queries()