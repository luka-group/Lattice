from preprocess_utils import get_highlighted_subtable


def linearize_table_with_index(table, cell_indices, table_page_title, table_section_title, order_cell=False):
    """
    Linearize the table based on highlighted cells and the table structure and return a string of its contents, including:
    * page title (<page_title>)
    * section title (<section_title>)
    * header (<header>)
    * highlighted cells (<cell>)
    [CLS] token & [SEP] token will be added after tokenization.
    Row and col ids starts from 1, 0 for metadata.
    Return char-level type/row/col ids for mapping after tokenization.
    If order_cell is True, cells in the linearized table will be in lexical order.
    """
    table_str = ""
    type_ids = []
    row_ids = []
    col_ids = []

    if table_page_title:
        table_str += "<page_title> " + table_page_title + " </page_title> "
        type_ids += [1 for _ in range(len(table_str))]
    if table_section_title:
        table_str += "<section_title> " + table_section_title + " </section_title> "
        type_ids += [2 for _ in range(len(table_str) - len(type_ids))]
    row_ids += [0 for _ in range(len(type_ids))]
    col_ids += [0 for _ in range(len(type_ids))]

    table_str += "<table> "
    type_ids += [3 for _ in range(len("<table> "))]
    row_ids += [0 for _ in range(len("<table> "))]
    col_ids += [0 for _ in range(len("<table> "))]

    subtable = (get_highlighted_subtable(
        table=table,
        cell_indices=cell_indices,
        with_heuristic_headers=True))

    if order_cell:
        subtable = sorted(subtable, key=lambda x: x["cell"]["value"])

    for item in subtable:
        cell = item["cell"]
        row_headers = item["row_headers"]
        col_headers = item["col_headers"]
        r_index = item["row_index"]
        c_index = item["col_index"]

        # The value of the cell.
        item_str = "<cell> " + cell["value"] + " "

        # All the headers associated with this cell.
        headers = col_headers + row_headers
        headers = sorted(headers, key=lambda x: x["value"])
        for header in headers:
            item_str += "<header> " + header["value"] + " </header> "

        item_str += "</cell> "
        cell_length = len(item_str)

        table_str += item_str
        type_ids += [3 for _ in range(cell_length)]
        row_ids += [r_index + 1 for _ in range(cell_length)]
        col_ids += [c_index + 1 for _ in range(cell_length)]

    table_str += "</table>"
    type_ids += [3 for _ in range(len("</table>"))]
    row_ids += [0 for _ in range(len("</table>"))]
    col_ids += [0 for _ in range(len("</table>"))]

    if cell_indices:
        assert "<cell>" in table_str
    assert len(table_str) == len(type_ids)
    assert len(row_ids) == len(type_ids)
    assert len(col_ids) == len(type_ids)

    return table_str, type_ids, row_ids, col_ids
