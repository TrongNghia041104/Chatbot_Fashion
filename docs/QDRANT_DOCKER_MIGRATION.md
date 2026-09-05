# Qdrant Docker Migration

This project keeps the new 65k embedded-local Qdrant data as the source:

```text
qdrant_data_65k/qdrant_data
```

Docker Qdrant server uses:

```text
storage/qdrant
```

Do not mount `qdrant_data_65k/qdrant_data` directly into Docker. Migrate the
collections into the running Qdrant server instead.

## 1. Start Qdrant Docker

```powershell
docker compose up -d qdrant
```

Check server:

```powershell
Invoke-RestMethod http://localhost:6333/collections
```

## 2. Migrate the 65k Data Into Docker Qdrant

Run from the project root:

```powershell
python scripts/migrate_qdrant_local_to_server.py --recreate
```

This copies these collections without embedding again:

```text
fashion_products_vifashionclip_vi_65k_structured_vi
fashion_products_fashionclip_image_main_65k
layer_b_female
layer_b_male
```

`--recreate` intentionally replaces existing target collections in
`storage/qdrant`.

## 3. Use Docker Qdrant at Runtime

The ViFashionCLIP notebook should connect with:

```python
client = QdrantClient(url="http://localhost:6333")
```

The embedded path remains only as backup/source:

```text
qdrant_data_65k/qdrant_data
```

## 4. Create Snapshots for Server Upload

After migration succeeds:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/create_qdrant_snapshots.ps1
```

The snapshot files are created inside Docker Qdrant storage and can be uploaded
to another Qdrant server.
