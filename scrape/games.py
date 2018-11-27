from bs4 import BeautifulSoup
from .common import simple_get

def parse_game(raw_html):
    try:
        html = BeautifulSoup(raw_html, 'html.parser') 

        header = html.find('h3',class_='page-title').get_text()
        game_name = header.strip().partition("\r")[0] #make sure it's a hex-13 game

        if game_name != "Hex-Size 13":
            return []
        end = header.partition('#')[2]
        gid = end[0:end.find('\r')]
        status = (end[end.find('\r'):]).strip()
        if status != "(game finished)":
            return []

        game = html.find_all('div', class_='portlet-body')[3] #for move list

        players = html.find_all('div', class_ = "col-xs-6 col-md-6")
        black = players[0].find('a').get_text()
        white = players[1].find('a').get_text()

        black_rating = players[0].find_all('br')[1].get_text().strip()
        white_rating = players[1].find_all('br')[1].get_text().strip()

        moves = game.find_all('b')
        move_list = []

        if moves[1].get_text() == "2.swap":
            m1 = moves[0].get_text().split(".",1)[1]
            move_list.append(m1+"*")
        else:
            move_list.append(moves[0].get_text().split(".",1)[1])
            move_list.append(moves[1].get_text().split(".",1)[1])

        turn = "black"
        for move in moves[2:]:
            m = move.get_text().split(".",1)[1]

            if m != "resign": # resign means they ended the game, not that the other player necessarily won
                move_list.append(m)

            if turn == "black":
                turn = "white"
            else: turn = "black"

        winner = turn #should check if they're the winner, the other resigned, too few moves, or the game is unfinished
        move_string = ''.join(move_list)
        return [gid,black,white,black_rating,white_rating,move_string,winner]
    except: 
        return []    