# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Baseline preprocessing utilities."""
import copy


def _add_adjusted_col_offsets(table):
    """Add adjusted column offsets to take into account multi-column cells."""
    adjusted_table = []
    for row in table:
        real_col_index = 0
        adjusted_row = []
        for cell in row:
            adjusted_cell = copy.deepcopy(cell)
            adjusted_cell["adjusted_col_start"] = real_col_index
            adjusted_cell["adjusted_col_end"] = (
                    adjusted_cell["adjusted_col_start"] + adjusted_cell["column_span"])
            real_col_index += adjusted_cell["column_span"]
            adjusted_row.append(adjusted_cell)
        adjusted_table.append(adjusted_row)
    return adjusted_table


def _get_heuristic_row_headers(adjusted_table, row_index, col_index):
    """Heuristic to find row headers."""
    row_headers = []
    row = adjusted_table[row_index]
    for i in range(0, col_index):
        if row[i]["is_header"]:
            row_headers.append(row[i])
    return row_headers


def _get_heuristic_col_headers(adjusted_table, row_index, col_index):
    """Heuristic to find column headers."""
    adjusted_cell = adjusted_table[row_index][col_index]
    adjusted_col_start = adjusted_cell["adjusted_col_start"]
    adjusted_col_end = adjusted_cell["adjusted_col_end"]
    col_headers = []
    for r in range(0, row_index):
        row = adjusted_table[r]
        for cell in row:
            if (cell["adjusted_col_start"] < adjusted_col_end and
                    cell["adjusted_col_end"] > adjusted_col_start):
                if cell["is_header"]:
                    col_headers.append(cell)

    return col_headers


def get_highlighted_subtable(table, cell_indices, with_heuristic_headers=False):
    """Extract out the highlighted part of a table."""
    highlighted_table = []

    adjusted_table = _add_adjusted_col_offsets(table)

    for (row_index, col_index) in cell_indices:
        cell = table[row_index][col_index]
        if with_heuristic_headers:
            row_headers = _get_heuristic_row_headers(adjusted_table, row_index,
                                                     col_index)
            col_headers = _get_heuristic_col_headers(adjusted_table, row_index,
                                                     col_index)
        else:
            row_headers = []
            col_headers = []

        highlighted_cell = {
            "cell": cell,
            "row_headers": row_headers,
            "col_headers": col_headers,
            "row_index": row_index,
            "col_index": col_index
        }
        highlighted_table.append(highlighted_cell)

    return highlighted_table

