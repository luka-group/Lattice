import sys
import csv
import jsonlines
from tqdm import tqdm
import argparse
 

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--json_file", required=True)
parser.add_argument("-o", "--csv_file", required=True)
args = parser.parse_args()

with jsonlines.open(args.json_file) as reader, open(args.csv_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["text", "summary", "type_ids", "row_ids", "col_ids"])
    for sample in tqdm(reader):
        src = sample["subtable_metadata_str"]
        if "sentence_annotations" in sample:
            tgt = sample["sentence_annotations"][0]["final_sentence"]
        else:
            tgt=' '
        type_ids = sample["type_ids"]
        row_ids = sample["row_ids"]
        col_ids = sample["col_ids"]
        writer.writerow([src, tgt, type_ids, row_ids, col_ids])

