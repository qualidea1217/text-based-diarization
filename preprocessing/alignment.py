import json

from align4d import align

# Load hypothesis
with open("/local/scratch/pwu54/Text-based SD Dataset/AMI/whisper_output/EN2001a.Mix-Headset.txt", 'r') as hypo_in:
    hypothesis = hypo_in.read()

# Load reference
with open("/local/scratch/pwu54/Text-based SD Dataset/AMI/transcript/EN2001a.json", 'r') as ref_in:
    reference = json.load(ref_in)

aligned_result = align.align(hypothesis, reference)
print(aligned_result)
