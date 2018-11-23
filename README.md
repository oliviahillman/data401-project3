# Hex AI Project

## Scraping

The web scraping code works by loading hard-coded tournament pages and scraping those pages for usable game ids. Those game ids are turned into urls and run through the game scraping which finally outputs information about the individual game. Our scraper returns the following fields:

- gid
- black
- white
- black_rating
- white_rating
- move_list
- winner

### Requirements

```
pip install aiohttp
```

`aiohttp` is used to perform asynchronous http requests.

### Running

```
./get_all_games.py
```

This usually takes about 15 minutes and produces `data/games.csv` which contains about 47,000 games at last run.s