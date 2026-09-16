import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPVisionModelWithProjection


TEACHER_MODEL_NAME = "patrickjohncyh/fashion-clip"
STUDENT_MODEL_NAME = "AITeamVN/Vietnamese_Embedding_v2"
STUDENT_MAX_LENGTH = 128
PROJECTION_HIDDEN_DIM = 1024
PROJECTION_NUM_LAYERS = 3
PROJECTION_DROPOUT = 0.05


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=3, dropout=0.05):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(0, num_layers - 2)):
            layers.append(ResidualMLPBlock(hidden_dim, dropout=dropout))
        layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


class Stage2StudentProjection(nn.Module):
    def __init__(self, encoder, projection_head):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs.last_hidden_state, attention_mask)
        return self.projection_head(pooled)


def resolve_image_path(image_root, raw_path):
    raw_path = str(raw_path)
    if os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path
    candidate = os.path.join(image_root, raw_path)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(image_root, os.path.basename(raw_path))


@torch.no_grad()
def encode_vi_texts(model, tokenizer, texts, device, batch_size):
    embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode Vietnamese text"):
        batch = [str(x) for x in texts[i : i + batch_size]]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=STUDENT_MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        embeds.append(F.normalize(out, p=2, dim=-1).cpu())
    return torch.cat(embeds, dim=0)


@torch.no_grad()
def encode_images(model, processor, image_paths, device, batch_size):
    embeds = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encode images"):
        paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        out = model(**inputs).image_embeds
        embeds.append(F.normalize(out, p=2, dim=-1).cpu())
    return torch.cat(embeds, dim=0)


def recall_metrics(text_embeds, image_embeds):
    sim = text_embeds @ image_embeds.T
    n = sim.shape[0]
    targets = torch.arange(n)
    t2i_sorted = sim.argsort(dim=1, descending=True)
    i2t_sorted = sim.T.argsort(dim=1, descending=True)
    t2i_ranks = (t2i_sorted == targets[:, None]).nonzero(as_tuple=True)[1] + 1
    i2t_ranks = (i2t_sorted == targets[:, None]).nonzero(as_tuple=True)[1] + 1
    metrics = {"pairs": n}
    for k in [1, 5, 10]:
        metrics[f"T2I_R@{k}"] = (t2i_ranks <= k).float().mean().item() * 100
        metrics[f"I2T_R@{k}"] = (i2t_ranks <= k).float().mean().item() * 100
    metrics["T2I_MedR"] = torch.median(t2i_ranks.float()).item()
    metrics["I2T_MedR"] = torch.median(i2t_ranks.float()).item()
    return metrics


def load_student_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    teacher_dim = int(checkpoint.get("teacher_dim", 512))
    encoder = AutoModel.from_pretrained(checkpoint.get("student_model_name", STUDENT_MODEL_NAME))
    projection = ProjectionHead(
        encoder.config.hidden_size,
        teacher_dim,
        hidden_dim=PROJECTION_HIDDEN_DIM,
        num_layers=PROJECTION_NUM_LAYERS,
        dropout=PROJECTION_DROPOUT,
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    projection.load_state_dict(checkpoint["projection_state_dict"])
    return Stage2StudentProjection(encoder, projection).to(device).eval()


def main():
    base_dir = Path(__file__).resolve().parent
    default_ckpt = base_dir / "vifashionclip_aiteamvn_embedding_v2_projection_336k" / "stage2_last_layers" / "best_stage2_model.pt"

    parser = argparse.ArgumentParser(description="Quick test Vietnamese FashionCLIP student checkpoint.")
    parser.add_argument("--csv", required=True, help="CSV with image_path and description_vi columns.")
    parser.add_argument("--image-root", default=".", help="Folder used to resolve relative image_path values.")
    parser.add_argument("--checkpoint", default=str(default_ckpt))
    parser.add_argument("--query", default=None, help="Vietnamese query for text-to-image search.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    required = {"image_path", "description_vi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    df["resolved_image_path"] = df["image_path"].apply(lambda p: resolve_image_path(args.image_root, p))
    df = df[df["resolved_image_path"].apply(os.path.exists)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid image paths found.")

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)
    text_model = load_student_model(args.checkpoint, device)
    vision_model = CLIPVisionModelWithProjection.from_pretrained(TEACHER_MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(TEACHER_MODEL_NAME)

    image_paths = df["resolved_image_path"].tolist()
    image_embeds = encode_images(vision_model, processor, image_paths, device, args.batch_size)

    if args.query:
        query_embed = encode_vi_texts(text_model, tokenizer, [args.query], device, args.batch_size)
        scores = (query_embed @ image_embeds.T).squeeze(0)
        top_scores, top_idx = torch.topk(scores, k=min(args.top_k, len(df)))
        print("\nTop results:")
        for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_idx.tolist()), start=1):
            row = df.iloc[idx]
            print(f"{rank}. score={score:.4f} | image={row['resolved_image_path']} | vi={row['description_vi']}")
    else:
        text_embeds = encode_vi_texts(text_model, tokenizer, df["description_vi"].tolist(), device, args.batch_size)
        metrics = recall_metrics(text_embeds, image_embeds)
        print("\nPaired retrieval metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
