import asyncio
import random
from datetime import timedelta

from crawlee import ConcurrencySettings, Request
from crawlee.crawlers import (
    PlaywrightCrawler,
    PlaywrightCrawlingContext,
)
from crawlee.sessions import SessionPool


async def human_delay(min_delay=1.0, max_delay=2.5):
    await asyncio.sleep(random.uniform(min_delay, max_delay))


async def main() -> None:
    crawler = PlaywrightCrawler(
        max_requests_per_crawl=100,
        headless=True,
        browser_type='firefox',
        max_session_rotations=5,
        concurrency_settings=ConcurrencySettings(
            max_concurrency=1,
            max_tasks_per_minute=10,
        ),
        session_pool=SessionPool(
            max_pool_size=1,
            create_session_settings={
                'max_usage_count': 250,
                'max_age': timedelta(hours=5),
                'max_error_score': 25,
            },
        ),
    )

    @crawler.router.default_handler
    async def request_handler(context: PlaywrightCrawlingContext) -> None:
        context.log.info(f'Processing {context.request.url} ...')

    @crawler.router.handler('home')
    async def login_handler(context: PlaywrightCrawlingContext) -> None:
        context.log.info(f'Processing login {context.request.url} ...')

        if not context.session:
            raise RuntimeError('Session not found')

        import random
        usernames = ['elonmusk', 'nasa', 'katyperry']
        username = random.choice(usernames)

        await context.page.type('.search-bar > form:nth-child(1) > input:nth-child(2)', username, delay=100)
        await context.page.click('span.icon-search')

        await context.page.wait_for_url(f'https://xcancel.com/search?f=tweets&q={username}')
        context.log.info('Search results loaded')
        await human_delay()

        await context.page.click('li.tab-item:nth-child(2) > a:nth-child(1)')
        context.log.info('Switched to tweets tab')
        await human_delay()

        await context.page.click(
            'div.timeline-container div.timeline div.timeline-item div.tweet-body.profile-result div.tweet-header div.tweet-name-row div.fullname-and-username a.fullname'
        )

        await context.page.wait_for_url(f'https://xcancel.com/{username}')
        await human_delay()

        all_texts = set()

        max_pages = 3
        current_page = 0

        while True and current_page < max_pages:
            current_page += 1

            tweets = await context.page.query_selector_all('.timeline-item')
            new_tweets_count = 0

            for tweet_el in tweets:
                retweet_el = await tweet_el.query_selector('div.retweet-header')
                if retweet_el:
                    continue

                content_el = await tweet_el.query_selector('div.tweet-content.media-body')
                if not content_el:
                    continue

                text = (await content_el.inner_text()).strip()
                if text not in all_texts:
                    all_texts.add(text)
                    new_tweets_count += 1

            context.log.info(f'Collected {new_tweets_count} new tweets (total: {len(all_texts)})')

            try:
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

            except Exception as e:
                context.log.warning(f'Pagination ended or error occurred: {e}')
                break

        print(f'Total tweets: {len(all_texts)}')
        print(list(all_texts))

    await crawler.run([Request.from_url('https://xcancel.com/', label='home')])


if __name__ == '__main__':
    asyncio.run(main())
