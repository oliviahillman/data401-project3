import csv
from bs4 import BeautifulSoup
from .common import simple_get
from pathlib import Path

NUM_RATING_GAMES = 7798
RATINGS_PAGE_TEMPLATE = ("http://ww.littlegolem.net"
                         "/jsp/games/gamedetail.jsp"
                         "?gtid=hex&page=rg&size={}")

GAME_URL_TEMPLATE = ("http://www.littlegolem.net"
                     "/jsp/game/game.jsp?gid={}")

DEFAULT_TOURNAMENT_TEMPLATE = ("http://www.littlegolem.net"
                               "/jsp/tournament/tournament.jsp"
                               "?trnid=hex.in.DEFAULT.{}")

CHAMPIONSHIP_TOURNAMENT_TEMPLATE = ("http://www.littlegolem.net"
                                    "/jsp/tournament/tournament.jsp"
                                    "?trnid=hex.ch.{}.{}.{}")

VALID_CHAMPIONSHIP_IDS_FILE = "valid_championship_ids.csv"
        
OLD_MONTHLY_TOURNAMENT_TEMPLATE = ("http://www.littlegolem.net"
                                   "/jsp/tournament/tournament.jsp"
                                   "?trnid=hex.mc.{}.{}.{}.{}")

MONTHLY_TOURNAMENT_TEMPLATE = ("http://www.littlegolem.net"
                               "/jsp/tournament/tournament.jsp"
                               "?trnid=hex.DEFAULT.mc.{}.{}.{}.{}")

VALID_MONTHLY_IDS_FILE = "valid_monthly_ids.csv"
        
VALID_OLD_MONTHLY_IDS_FILE = "valid_old_monthly_ids.csv"

USER_TOURNAMENTS = [
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.7.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.7.1.2",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.7.1.3",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.7.2.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.233.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.252.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.252.1.2",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.252.1.3",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.252.1.4",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.252.1.5",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.252.1.6",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.252.1.7",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.252.2.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.252.2.2",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.252.2.3",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.252.3.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.282.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.282.1.2",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.282.1.3",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.282.2.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.461.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.528.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.599.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.600.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.599.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.606.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.619.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.654.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.654.1.2",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.654.1.3",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.654.2.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.665.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.677.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.707.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.811.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.811.1.2",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.811.1.3",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.811.2.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.960.1.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.960.1.2",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.960.1.3",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.960.2.1",
    "http://www.littlegolem.net/jsp/tournament/tournament.jsp?trnid=ut.hex.1109.1.1"
]

def parse_tournament_table(raw_html):
    """
    Parse a tournament table for a list of game IDs.
    
    Also works for the rating games page
    """
    
    try:
        html = BeautifulSoup(raw_html, 'html.parser')
        
        game_links = html.select('.portlet-body > table tr > td:nth-of-type(1) > b > a')
        
        return list(map(lambda link: link.get_text().replace('#', ''), game_links))
    except Exception as e:
        print(e)
        return []
    
def append_from_file(to_append, filename, f):
    with open(filename, newline='') as ids_file:
        reader = csv.DictReader(ids_file)
        for row in reader:
            to_append.append(
                f(row)
            )
    
def get_all_tournament_urls(data_dir):
    from os import getcwd

    data_root = Path(data_dir)
    
    tournaments = [
        RATINGS_PAGE_TEMPLATE.format(NUM_RATING_GAMES)
    ]
    
    for i in range(1, 162):
        tournaments.append(
            DEFAULT_TOURNAMENT_TEMPLATE.format(i)
        )
        
    append_from_file(tournaments, data_root / VALID_MONTHLY_IDS_FILE,
                    lambda row: MONTHLY_TOURNAMENT_TEMPLATE.format(row['year'],
                                                                   row['month'],
                                                                   row['j'], row['k']))
    append_from_file(tournaments, data_root / VALID_CHAMPIONSHIP_IDS_FILE,
                    lambda row: CHAMPIONSHIP_TOURNAMENT_TEMPLATE.format(row['i'], row['j'], row['k']))

    append_from_file(tournaments, data_root / VALID_OLD_MONTHLY_IDS_FILE,
                    lambda row: OLD_MONTHLY_TOURNAMENT_TEMPLATE.format(row['year'],
                                                                   row['month'],
                                                                   row['j'], row['k']))

    tournaments.extend(USER_TOURNAMENTS)
    
    return tournaments