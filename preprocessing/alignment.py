import csv
import json

from align4d import align

# Load hypothesis
with open("/local/scratch/pwu54/Text-based SD Dataset/AMI/whisper_output/EN2001a.Mix-Headset.txt", 'r') as hypo_in:
    hypothesis = hypo_in.read()

# Load reference
with open("/local/scratch/pwu54/Text-based SD Dataset/AMI/transcript/EN2001a.json", 'r') as ref_in:
    reference = json.load(ref_in)

aligned_result = align.align(hypothesis, reference)

with open("EN2001a.csv", 'w') as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(aligned_result["hypothesis"])
    for key, value in aligned_result["reference"].items():
        print(key)
        writer.writerow(value)
