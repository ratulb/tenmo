import urllib.request
from html.parser import HTMLParser

#https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub

class _TableParser(HTMLParser):
    """Minimal parser that collects text from the first <table>'s <td> cells."""

    def __init__(self):
        super().__init__()
        self._in_table = False
        self._in_tr = False
        self._in_td = False
        self._rows = []
        self._cur_row = []
        self._cur_cell = []
        self._tables_found = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._tables_found += 1
            self._in_table = True
            self._rows = []
        elif tag == "tr" and self._in_table:
            self._in_tr = True
            self._cur_row = []
        elif tag in ("td", "th") and self._in_tr:
            self._in_td = True
            self._cur_cell = []

    def handle_endtag(self, tag):
        if tag == "table":
            self._in_table = False
        elif tag == "tr" and self._in_tr:
            self._in_tr = False
            self._rows.append(self._cur_row)
            self._cur_row = []
        elif tag in ("td", "th") and self._in_td:
            self._in_td = False
            self._cur_row.append("".join(self._cur_cell).strip())
            self._cur_cell = []

    def handle_data(self, data):
        if self._in_td:
            self._cur_cell.append(data)

    def handle_entityref(self, name):
        if self._in_td:
            self._cur_cell.append(f"&{name};")

    def handle_charref(self, name):
        if self._in_td:
            self._cur_cell.append(f"&#{name};")


def decode(url: str) -> None:
    """Fetch a published Google Doc, parse the character grid, and print it."""
    with urllib.request.urlopen(url) as resp:
        html = resp.read().decode("utf-8")

    parser = _TableParser()
    parser.feed(html)

    if not parser._rows:
        print("No table found.")
        return

    points = []
    for row in parser._rows[1:]:
        if len(row) < 3:
            continue
        try:
            x = int(row[0])
            ch = row[1]
            y = int(row[2])
        except ValueError:
            continue
        points.append((x, y, ch))

    if not points:
        print("No data found.")
        return

    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)

    grid = [[" "] * (max_x + 1) for _ in range(max_y + 1)]
    for x, y, ch in points:
        grid[y][x] = ch

    for row in grid:
        print("".join(row))
