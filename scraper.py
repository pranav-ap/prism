import asyncio
import json
import re
import random
from dataclasses import asdict
from datetime import timedelta
from typing import Set

from crawlee import ConcurrencySettings, Request
from crawlee.configuration import Configuration
from crawlee.crawlers import (
    PlaywrightCrawler,
    PlaywrightCrawlingContext,
)
from crawlee.sessions import SessionPool

from data_source import ScraperTweet


async def human_delay(min_delay=1.0, max_delay=2.5):
    await asyncio.sleep(random.uniform(min_delay, max_delay))


def process_text(text: str) -> str:
    # Remove all emojis and symbols except for basic punctuation
    cleaned = re.sub(r'[^\w\s,.!?@#:/-]', '', text, flags=re.UNICODE)
    cleaned = cleaned.strip()
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


async def tweet_extraction_handler(context: PlaywrightCrawlingContext) -> Set[ScraperTweet]:
    username = context.request.user_data.get('username', '')

    if not username:
        raise ValueError('Username is required in user_data')

    extracted_tweets = set()

    await context.page.get_by_placeholder('Search...').type(username, delay=100)
    await context.page.click('span.icon-search')

    await context.page.wait_for_selector('li.tab-item:nth-child(2) > a:nth-child(1)')
    print('Search results loaded')
    await human_delay()

    await context.page.get_by_text('Users').click(timeout=5000)
    print('Switched to Users tab')
    await human_delay()

    link = context.page.locator(f'a.username[title="@{username}"]', has_text=f'@{username}').first

    if await link.count() == 0:
        raise ValueError(f'No link found for @{username}')

    print(f'Found link for @{username}')
    await link.click(timeout=2000)
    await human_delay()

    MAX_PAGES = context.request.user_data.get('max_pages', 3)
    current_page = 0

    while True and current_page < MAX_PAGES:
        current_page += 1
        new_tweets_count = 0

        tweets = await context.page.query_selector_all('.timeline-item')

        for tweet_el in tweets:
            if await tweet_el.query_selector('div.retweet-header') or await tweet_el.query_selector('div.pinned'):
                continue

            content_el, time_el = await asyncio.gather(
                tweet_el.query_selector('div.tweet-content.media-body'),
                tweet_el.query_selector('div.tweet-name-row span.tweet-date a')
            )

            if not content_el or not time_el:
                continue

            time = await time_el.get_attribute('title')

            text = await content_el.inner_text()
            text = process_text(text)

            if not len(text):
                continue

            extracted_tweet = ScraperTweet(
                id=len(extracted_tweets),
                text=text,
                timestamp=time,
            )

            extracted_tweets.add(extracted_tweet)
            new_tweets_count += 1

        print(f'Collected {new_tweets_count} new tweets (total: {len(extracted_tweets)})')

        show_more = await context.page.query_selector('div.show-more a')
        if not show_more:
            print('No more "Show more" button found.')
            break

        await show_more.click()
        print('Clicked "Show more"...')
        await human_delay()

        await context.page.wait_for_selector('.timeline-item', timeout=5000)

        if new_tweets_count == 0:
            print('No new tweets found after clicking "Show more".')
            break

    print(f'Total tweets: {len(extracted_tweets)}')

    return extracted_tweets


def setup_crawler() -> PlaywrightCrawler:
    crawler = PlaywrightCrawler(
        max_requests_per_crawl=1000,
        headless=True,
        # headless=False,
        browser_type='firefox',
        max_session_rotations=1,
        concurrency_settings=ConcurrencySettings(
            max_concurrency=1,
            max_tasks_per_minute=8,
        ),
        session_pool=SessionPool(
            max_pool_size=1,
            create_session_settings={
                'max_usage_count': 800,
                'max_age': timedelta(hours=2),
                'max_error_score': 50,
            },
        ),
        use_incognito_pages=True,
        configuration=Configuration(
            log_level='INFO',  # INFO CRITICAL
        )
    )

    @crawler.router.default_handler
    async def request_handler(context: PlaywrightCrawlingContext) -> None:
        context.log.info(f'Processing {context.request.url} ...')

    @crawler.router.handler('home')
    async def home_handler(context: PlaywrightCrawlingContext) -> None:
        context.log.info(f'Processing home {context.request.url} ...')

        extracted_tweets = await tweet_extraction_handler(context)
        extracted_tweets = [asdict(t) for t in extracted_tweets]

        username = context.request.user_data.get('username', '')

        if not username:
            raise ValueError('Username is required in user_data')

        with open(f'output/extracted_tweets_{username}.json', 'w', encoding='utf-8') as f:
            json.dump(extracted_tweets, f, ensure_ascii=True, indent=2)

    return crawler


async def main() -> None:
    crawler: PlaywrightCrawler = setup_crawler()

    usernames = [
        # 'krystalball',
        'nytimes',
        # 'nytimeses'
    ]

    try:
        for username in usernames:
            request = Request.from_url(
                url='https://xcancel.com/',
                label='home',
                userData={
                    'username': username,
                    'max_pages': 5,
                }
            )

            print(f'Starting Crawler for username: {username}')
            await crawler.run([request])
            print(f'Crawling finished for username: {username}')

        print('Crawling Finished Successfully!')

    except Exception as e:
        print(f'Crawler Failed!')
        print(e)


if __name__ == '__main__':
    asyncio.run(main())
