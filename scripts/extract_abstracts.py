import csv
import sys

if len(sys.argv) < 2:
    print("Usage: python extract_abstracts.py <input_csv_file>")
    sys.exit(1)

input_csv = sys.argv[1]

# Output files
train_file = "text2cad_rft_train.txt"
test_file = "text2cad_rft_test.txt"
val_file = "text2cad_rft_val.txt"

with open(input_csv, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    
    with open(train_file, "w", encoding='utf-8') as train_out, \
         open(test_file, "w", encoding='utf-8') as test_out, \
         open(val_file, "w", encoding='utf-8') as val_out:
        
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue  # skip malformed rows
            
            abstract = row[1].strip()
            if not abstract:
                continue  # skip empty abstracts

            # Pattern: 3 train, 1 test, 1 val, then repeat
            mod = i % 5
            if mod in (0, 1, 2):
                train_out.write(abstract + "\n")
            elif mod == 3:
                test_out.write(abstract + "\n")
            else:
                val_out.write(abstract + "\n")

print("✅ Done! Extracted abstracts written to:")
print("  - text2cad_rft_train.txt (3/5 of rows)")
print("  - text2cad_rft_test.txt  (1/5 of rows)")
print("  - text2cad_rft_val.txt   (1/5 of rows)")

