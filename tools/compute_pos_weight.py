import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import deep_stream
from pathlib import Path

def main():
    output_root = Path('output')
    gt_root = Path('data') / 'redkitchen_edge_gt'
    seqs = ['seq-01']
    samples = deep_stream._discover_labeled_samples(Path('output'), gt_root, seqs)
    if not samples:
        print('No labeled training samples found for', seqs)
        return
    w = deep_stream._count_positive_weight(samples)
    print(w)

if __name__ == '__main__':
    main()
