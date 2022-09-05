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
"""Augments json files with table linearization used by baselines.

Note that this code is merely meant to be starting point for research and
there may be much better table representations for this task.
"""
import copy
import json
from tqdm import tqdm
from absl import app
from absl import flags
import six

from table_linearization import linearize_table_with_index

flags.DEFINE_string("input_path", None, "Input json file.")

flags.DEFINE_string("output_path", None, "Output directory.")

flags.DEFINE_integer("examples_to_visualize", 100,
                     "Number of examples to visualize.")

FLAGS = flags.FLAGS


def _generate_processed_examples(input_path):
  processed_json_examples = []
  with open(input_path, "r", encoding="utf-8") as input_file:
    for line in tqdm(input_file):
      line = six.ensure_text(line, "utf-8")
      json_example = json.loads(line)
      table = json_example["table"]
      table_page_title = json_example["table_page_title"]
      table_section_title = json_example["table_section_title"]
      cell_indices = json_example["highlighted_cells"]

      table_metadata_str, type_ids, row_ids, col_ids = (
          linearize_table_with_index(
              table=table,
              cell_indices=cell_indices,
              table_page_title=table_page_title,
              table_section_title=table_section_title,
              order_cell=True))

      processed_json_example = copy.deepcopy(json_example)
      processed_json_example["subtable_metadata_str"] = table_metadata_str
      processed_json_example["type_ids"] = " ".join([str(x) for x in type_ids])
      processed_json_example["row_ids"] = " ".join([str(x) for x in row_ids])
      processed_json_example["col_ids"] = " ".join([str(x) for x in col_ids])
      processed_json_examples.append(processed_json_example)

  print("Num examples processed: %d" % len(processed_json_examples))
  return processed_json_examples


def main(_):
  input_path = FLAGS.input_path
  output_path = FLAGS.output_path
  processed_json_examples = _generate_processed_examples(input_path)
  with open(output_path, "w", encoding="utf-8") as f:
    for json_example in processed_json_examples:
      f.write(json.dumps(json_example) + "\n")


if __name__ == "__main__":
  app.run(main)
