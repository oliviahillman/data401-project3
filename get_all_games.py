#!/usr/bin/env python

from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from scrape.tournaments import get_all_tournament_urls, parse_tournament_table
from scrape.games import parse_game
from os.path import join
from time import time
import asyncio as aio
import csv
import aiohttp
from contextlib import closing

DATA_DIRECTORY = 'data/'
OUTPUT_CSV_HEADER = ["gid", "black", "white", "black_rating",
                     "white_rating","move_list","winner"]

NUM_CPUS = cpu_count()
NUM_CPU_WORKERS = NUM_CPUS - 1

GAME_URL_TEMPLATE = "http://littlegolem.net/jsp/game/game.jsp?gid={}"

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def process_game(session, executor, game_id):
    game_url = GAME_URL_TEMPLATE.format(game_id)
    raw_html = await fetch(session, game_url)
    loop = aio.get_event_loop()
    game = await loop.run_in_executor(executor, parse_game, raw_html)

    return game

async def process_tournament(session, executor, url):
    raw_html = await fetch(session, url)
    loop = aio.get_event_loop()
    game_ids = await loop.run_in_executor(executor, parse_tournament_table, raw_html)

    results = await aio.gather(
        *[ process_game(session, executor, game_id) for game_id in game_ids ]
    )

    return results

async def scrape_pipeline(executor, csv_writer, tournament_urls):
    async with aiohttp.ClientSession() as session:
        results = await aio.gather(
            *[ process_tournament(session, executor, url) for url in tournament_urls ]
        )

        games = (game for sublist in results for game in sublist)

        total_game_count = 0
        valid_game_count = 0
        for game in games:
            total_game_count += 1
            if game != []:
                valid_game_count += 1
                csv_writer.writerow(game)
        print(("Scraped {} valid hex games"
               " out of {} total games.").format(valid_game_count, total_game_count))

def main():
    output_path = join(DATA_DIRECTORY, 'games.csv')
    tournament_urls = get_all_tournament_urls(DATA_DIRECTORY)
    print("Loaded {} tournament urls.".format(len(tournament_urls)))

    loop = aio.get_event_loop()
    with ProcessPoolExecutor(max_workers=NUM_CPU_WORKERS) as executor, \
         open(output_path, 'w') as output_file:

        writer = csv.writer(output_file)
        writer.writerow(OUTPUT_CSV_HEADER)

        loop.run_until_complete(scrape_pipeline(executor, writer, tournament_urls))
    loop.close()

if __name__ == "__main__":
    main()
