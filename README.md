Project Memory Game

Auteur: Robbe Vermeire & Joren Vandekerckhove
Datum: 14/12/2020

--------------------------------------------------------------------------------

OS: Windows 10

Versie controle:
    Python 3.8
    Opencv 4.4.0.46
    Numpy 1.19.1
    imutils 0.5.3
    psutil 5.7.3

IDE: Visual Code

--------------------------------------------------------------------------------

Bestanden/Mappen:

main.py
    python script dat het memory spel speelt via een video dat meegegeven is via de command line

helper_methods.py
    python script met hulp functies die zorgen voor de live feed, het zoeken achter paren, ...

grid.py
    klasse grid wordt hierin beschreven

tile.py
    klasse tile wordt hierin beschreven

result.png
    de afbeelding dat gecreëerd wordt achter dat het programma klaar is

commands.txt: commando's die werken met de video's

start commando example: python main.py --INDEX_VIDEO 6 --THRESHOLD_HAND 0.45 --THRESHOLD_SUB 10 --BORDER 30 --METHOD sub --WTA_K 3 --BORDER_ORB 31 --BORDER_IMG 0 --PLAY 0 --ROWS 4  --COLS 4 --SAVE 0

--------------------------------------------------------------------------------

Uitleg programma:

1. De video wordt overlopen en de belangrijke frames worden opgeslagen nadat een hand is verschenen.
2. Alle frames worden overlopen en er wordt gekeken waar er tegels omgedraaid zijn. Deze worden opgeslagen in een grid.
3. De grid wordt overlopen en bekeken waar er tegels met elkaar overeenkomen.
4. Als laatstes wordt een afbeelding getoond aan de gebruiker van het spel met alle tegels omgedraaid en de paren die bij elkaar horen.