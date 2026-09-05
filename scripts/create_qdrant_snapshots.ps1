param(
    [string]$QdrantUrl = "http://localhost:6333"
)

$collections = @(
    "fashion_products_vifashionclip_vi_65k_structured_vi",
    "fashion_products_fashionclip_image_main_65k",
    "layer_b_female",
    "layer_b_male"
)

foreach ($collection in $collections) {
    Write-Host "Creating snapshot for $collection ..."
    Invoke-RestMethod -Method Post -Uri "$QdrantUrl/collections/$collection/snapshots"
}
