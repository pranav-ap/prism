import asyncio
import json
import re
import random
from dataclasses import asdict
from datetime import timedelta

from crawlee import ConcurrencySettings, Request
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


async def main() -> None:
    crawler = PlaywrightCrawler(
        max_requests_per_crawl=500,
        headless=True,
        browser_type='firefox',
        max_session_rotations=1,
        concurrency_settings=ConcurrencySettings(
            max_concurrency=1,
            max_tasks_per_minute=10,
        ),
        session_pool=SessionPool(
            max_pool_size=1,
            create_session_settings={
                'max_usage_count': 500,
                'max_age': timedelta(hours=2),
                'max_error_score': 50,
            },
        ),
    )

    @crawler.router.default_handler
    async def request_handler(context: PlaywrightCrawlingContext) -> None:
        context.log.info(f'Processing {context.request.url} ...')

    @crawler.router.handler('home')
    async def home_handler(context: PlaywrightCrawlingContext) -> None:
        context.log.info(f'Processing home {context.request.url} ...')

        if not context.session:
            raise RuntimeError('Session not found')

        usernames = ['nytimes', 'techcrunch']
        username = random.choice(usernames)

        await context.page.type('.search-bar > form:nth-child(1) > input:nth-child(2)', username, delay=100)
        await context.page.click('span.icon-search')

        await context.page.wait_for_selector('li.tab-item:nth-child(2) > a:nth-child(1)')
        context.log.info('Search results loaded')
        await human_delay()

        await context.page.click('li.tab-item:nth-child(2) > a:nth-child(1)')
        context.log.info('Switched to tweets tab')
        await human_delay()

        await context.page.wait_for_selector('div.timeline div.timeline-item')
        context.log.info('Users Loaded')
        await human_delay()

        await context.page.click(
            'div.timeline div.timeline-item div.tweet-body.profile-result div.tweet-header div.tweet-name-row div.fullname-and-username a.fullname')
        await context.page.wait_for_selector('.timeline-item', timeout=5000)
        await human_delay()

        extracted_tweets = set()

        MAX_PAGES = 5
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

            context.log.info(f'Collected {new_tweets_count} new tweets (total: {len(extracted_tweets)})')

            show_more = await context.page.query_selector('div.show-more a')
            if not show_more:
                context.log.info('No more "Show more" button found.')
                break

            await show_more.click()
            context.log.info('Clicked "Show more"...')
            await human_delay()

            await context.page.wait_for_selector('.timeline-item', timeout=5000)

            if new_tweets_count == 0:
                context.log.info('No new tweets found after clicking "Show more".')
                break

        context.log.info(f'Total tweets: {len(extracted_tweets)}')
        extracted_tweets = [asdict(t) for t in extracted_tweets]

        with open('data/extracted_tweets.json', 'w', encoding='utf-8') as f:
            json.dump(extracted_tweets, f, ensure_ascii=False, indent=2)

    try:
        await crawler.run([Request.from_url('https://xcancel.com/', label='home')])
        print('Crawler finished successfully.')
    except Exception as e:
        print(f'Crawler failed: {e}')


if __name__ == '__main__':
    asyncio.run(main())
