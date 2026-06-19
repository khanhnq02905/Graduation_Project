import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import deep_stream

PROJECT_ROOT = Path(__file__).resolve().parents[1]
checkpoint = PROJECT_ROOT / 'results' / 'deep_stream' / 'deep_stream.pt'
out_dir = PROJECT_ROOT / 'results' / 'hard_negatives'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# load model
model = deep_stream.PointEdgeNet(input_dim=4, hidden_dim=256, dropout=0.2).to(device)
if not checkpoint.exists():
    raise SystemExit('Checkpoint not found: ' + str(checkpoint))
deep_stream._load_checkpoint(checkpoint, model, device)
model.eval()

# find train samples (use default train-seqs=seq-01)
train_seqs = ['seq-01']
samples = deep_stream._discover_labeled_samples(PROJECT_ROOT / 'output', PROJECT_ROOT / 'data' / 'redkitchen_edge_gt', train_seqs)
if not samples:
    raise SystemExit('No train samples')

for sample in samples:
    pcd = deep_stream.o3d.io.read_point_cloud(str(sample.ply_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        continue
    features = deep_stream._build_features(points)
    labels = np.load(sample.mask_path).astype(np.bool_)
    neg_idx = np.flatnonzero(~labels)
    if len(neg_idx) == 0:
        continue
    # run model on full frame in reasonably sized batches
    batch_size = 16384
    probs = np.zeros(len(features), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            chunk = torch.from_numpy(features[i:i+batch_size]).to(device)
            logits = model(chunk, None)
            if logits.dim() == 2:
                logits = logits.squeeze(0)
            probs[i:i+len(logits)] = torch.sigmoid(logits).cpu().numpy()
    neg_probs = probs[neg_idx]
    # select top-k negatives by probability
    k = min(len(neg_idx), 2048)
    top_inds = np.argsort(-neg_probs)[:k]
    selected = neg_idx[top_inds]
    out_path = out_dir / sample.seq_id
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / f"{sample.frame_name}_hard_neg.npy", selected.astype(np.int32))
    print('Saved hard negatives for', sample.seq_id, sample.frame_name, 'count', len(selected))

print('Done')
