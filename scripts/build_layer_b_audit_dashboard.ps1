param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

if (-not $OutputPath) {
    $OutputPath = Join-Path $ProjectRoot "outputs\layer_b_audit\Layer_B_Audit_Dashboard.html"
}

$layerDir = Join-Path $ProjectRoot "data\LayerB"
$stylistDir = Join-Path $ProjectRoot "data\stylists"
$notebookPath = Join-Path $layerDir "Build_Layer_B.ipynb"

function Html([object]$Value) {
    if ($null -eq $Value) { return "" }
    return [System.Net.WebUtility]::HtmlEncode([string]$Value)
}

function Pct([double]$Numerator, [double]$Denominator, [int]$Digits = 1) {
    if ($Denominator -eq 0) { return "0%" }
    return ((100 * $Numerator / $Denominator).ToString("N$Digits") + "%")
}

function Mean([double[]]$Values) {
    if (-not $Values -or $Values.Count -eq 0) { return 0 }
    return ($Values | Measure-Object -Average).Average
}

function Median([double[]]$Values) {
    if (-not $Values -or $Values.Count -eq 0) { return 0 }
    $s = @($Values | Sort-Object)
    $mid = [math]::Floor($s.Count / 2)
    if ($s.Count % 2 -eq 1) { return $s[$mid] }
    return ($s[$mid - 1] + $s[$mid]) / 2
}

function Percentile([double[]]$Values, [double]$P) {
    if (-not $Values -or $Values.Count -eq 0) { return 0 }
    $s = @($Values | Sort-Object)
    $idx = [math]::Ceiling($P * $s.Count) - 1
    $idx = [math]::Max(0, [math]::Min($idx, $s.Count - 1))
    return $s[$idx]
}

function CountValues($Rows, [string]$Column, [int]$Limit = 0) {
    $groups = @($Rows | ForEach-Object { $_.$Column } |
        Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } |
        Group-Object | Sort-Object -Property @{Expression='Count';Descending=$true}, @{Expression='Name';Descending=$false})
    if ($Limit -gt 0) { $groups = @($groups | Select-Object -First $Limit) }
    return @($groups | ForEach-Object {
        [pscustomobject]@{ label = [string]$_.Name; count = [int]$_.Count }
    })
}

function MakeOrdinalSet {
    return ,([System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal))
}

function MakeOrdinalDict {
    return ,([System.Collections.Generic.Dictionary[string,object]]::new([System.StringComparer]::Ordinal))
}

function ParseItemIds([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) { return @() }
    return @(($Raw -replace '，', ',') -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function ExtractOutfitStyle([string]$Description) {
    if ($Description -match '^([A-Za-z][A-Za-z\s\-/&]+?)\s+style[:\s]') {
        return $Matches[1].Trim()
    }
    return "Casual"
}

$categoryVi = @{
    "inner_top" = "Áo mặc trong (áo thun/sơ mi)"
    "mid_layer_top" = "Áo khoác nhẹ/Áo len"
    "outerwear" = "Áo khoác ngoài"
    "bottom" = "Quần/Chân váy"
    "onepiece" = "Đầm/Jumpsuit"
    "shoes" = "Giày dép"
    "bag" = "Túi xách"
    "accessory" = "Phụ kiện"
}

$seasonVi = @{
    "Spring" = "Mùa xuân"; "Summer" = "Mùa hè"; "Autumn" = "Mùa thu"; "Winter" = "Mùa đông"
    "Spring/Summer" = "Xuân hè"; "Autumn/Winter" = "Thu đông"
}
$occasionVi = @{
    "School" = "Học đường"; "Travel" = "Du lịch"; "Sports" = "Thể thao"; "Home" = "Ở nhà"
    "Social" = "Xã hội"; "Work" = "Công sở"; "Business" = "Công sở"; "Daily" = "Hàng ngày"
}

$maleItems = @(Import-Csv -LiteralPath (Join-Path $layerDir "label_en.csv") -Encoding UTF8)
$femaleItems = @(Import-Csv -LiteralPath (Join-Path $layerDir "label_en_Female.csv") -Encoding UTF8)
$maleLooks = @(Import-Csv -LiteralPath (Join-Path $layerDir "look_en.csv") -Encoding UTF8)
$femaleLooks = @(Import-Csv -LiteralPath (Join-Path $layerDir "look_en_female.csv") -Encoding UTF8)
$maleRulesRaw = Get-Content -Raw -Encoding UTF8 (Join-Path $stylistDir "Layer_B_Male_Knowledge.json") | ConvertFrom-Json
$femaleRulesRaw = Get-Content -Raw -Encoding UTF8 (Join-Path $stylistDir "Layer_B_Female_Knowledge.json") | ConvertFrom-Json
$maleRules = @($maleRulesRaw)
$femaleRules = @($femaleRulesRaw)
$notebook = Get-Content -Raw -Encoding UTF8 $notebookPath | ConvertFrom-Json

function GetDictKeys([string]$Code, [string]$DictName, [string]$EndFunction) {
    $start = $Code.IndexOf("$DictName = {")
    if ($start -lt 0) { return @() }
    $end = $Code.IndexOf("def $EndFunction", $start)
    if ($end -lt 0) { $end = $Code.Length }
    $section = $Code.Substring($start, $end - $start)
    $matches = [regex]::Matches($section, '(?m)^\s{4}"([^"]+)"\s*:\s*\[')
    return @($matches | ForEach-Object { $_.Groups[1].Value })
}

function MappingStatus([string]$Value, [string[]]$Keys) {
    $v = ([string]$Value).ToLowerInvariant().Trim()
    if ($Keys -contains $v) { return "Exact" }
    foreach ($key in @($Keys | Sort-Object Length -Descending)) {
        if ($v.Contains($key)) { return "Partial" }
    }
    return "Default"
}

$maleCode = ($notebook.cells[2].source -join "")
$femaleCode = ($notebook.cells[3].source -join "")
$bodyKeysMale = @(GetDictKeys $maleCode "OUTLINE_TO_BODY" "map_body")
$colorKeysMale = @(GetDictKeys $maleCode "COLOR_TO_SKIN" "map_skin")
$bodyKeysFemale = @(GetDictKeys $femaleCode "OUTLINE_TO_BODY" "map_body")
$colorKeysFemale = @(GetDictKeys $femaleCode "COLOR_TO_SKIN" "map_skin")

function AnalyzeMappingCoverage($Items, $BodyKeys, $ColorKeys) {
    $body = @($Items | ForEach-Object { MappingStatus $_.outline $BodyKeys } | Group-Object)
    $tone = @($Items | ForEach-Object { MappingStatus $_.color $ColorKeys } | Group-Object)
    function G($Groups, [string]$Name) { $x = $Groups | Where-Object Name -eq $Name | Select-Object -First 1; if ($x) { return $x.Count }; return 0 }
    return [pscustomobject]@{
        body_exact = G $body "Exact"; body_partial = G $body "Partial"; body_default = G $body "Default"
        tone_exact = G $tone "Exact"; tone_partial = G $tone "Partial"; tone_default = G $tone "Default"
    }
}

$maleCoverage = AnalyzeMappingCoverage $maleItems $bodyKeysMale $colorKeysMale
$femaleCoverage = AnalyzeMappingCoverage $femaleItems $bodyKeysFemale $colorKeysFemale

function AnalyzePair([string]$Name, $Items, $Looks, $Rules) {
    $itemById = MakeOrdinalDict
    foreach ($item in $Items) { $itemById[$item.itemID] = $item }

    $refCounts = MakeOrdinalDict
    $missingRefs = New-Object System.Collections.Generic.List[string]
    $itemsPerLook = New-Object System.Collections.Generic.List[double]
    $unicodeCommaLooks = 0
    $styleMatched = 0
    foreach ($look in $Looks) {
        if ($look.items.Contains('，')) { $unicodeCommaLooks++ }
        $ids = @(ParseItemIds $look.items)
        $itemsPerLook.Add($ids.Count)
        if ((ExtractOutfitStyle $look.look) -ne "Casual") { $styleMatched++ }
        foreach ($id in $ids) {
            if ($refCounts.ContainsKey($id)) { $refCounts[$id] = [int]$refCounts[$id] + 1 } else { $refCounts[$id] = 1 }
            if (-not $itemById.ContainsKey($id)) { $missingRefs.Add($id) }
        }
    }

    $orphans = @($Items | Where-Object { -not $refCounts.ContainsKey($_.itemID) } | ForEach-Object { $_.itemID })
    $reused = @($refCounts.GetEnumerator() | Where-Object { $_.Value -gt 1 } | Sort-Object Value -Descending | ForEach-Object {
        [pscustomobject]@{ itemID = $_.Key; appearances = $_.Value }
    })
    $dupItemIds = @($Items | Group-Object itemID | Where-Object Count -gt 1)
    $dupOutfitIds = @($Looks | Group-Object outfitID | Where-Object Count -gt 1)

    $ruleExact = MakeOrdinalSet
    foreach ($rule in $Rules) { [void]$ruleExact.Add([string]$rule.rule_key) }
    $ruleCaseInsensitive = @($Rules | Group-Object { ([string]$_.rule_key).ToLowerInvariant() } | Where-Object Count -gt 1)
    $reasonGroups = @($Rules | Group-Object ly_do_tu_van | Where-Object Count -gt 1)
    $reasonLengths = @($Rules | ForEach-Object { [double]([string]$_.ly_do_tu_van).Length })

    $allowedBody = @("Dáng quả lê", "Dáng quả táo", "Dáng đồng hồ cát", "Dáng chữ nhật", "Dáng cân đối", "Người thấp bé", "Người ngoại cỡ", "Người mảnh", "Mọi vóc dáng")
    $allowedTone = @("Da sáng", "Da trung bình", "Da ngăm", "Da ấm", "Mọi tone da")
    $bodyValues = @($Rules | ForEach-Object { @($_.dang_nguoi) } | ForEach-Object { $_ } | Where-Object { $_ })
    $toneValues = @($Rules | ForEach-Object { @($_.tone_da) } | ForEach-Object { $_ } | Where-Object { $_ })
    $badBody = @($bodyValues | Where-Object { $_ -notin $allowedBody })
    $badTone = @($toneValues | Where-Object { $_ -notin $allowedTone })

    $blankByField = @()
    foreach ($field in @("rule_key", "phong_cach", "boi_canh", "dang_nguoi", "tone_da", "goi_y_phoi_cung", "ly_do_tu_van")) {
        $blank = @($Rules | Where-Object {
            $v = $_.$field
            $null -eq $v -or @($v).Count -eq 0 -or [string]::IsNullOrWhiteSpace([string]$v)
        }).Count
        $blankByField += [pscustomobject]@{ field = $field; blank = $blank }
    }

    return [pscustomobject]@{
        name = $Name; item_count = $Items.Count; look_count = $Looks.Count; rule_count = $Rules.Count
        reference_count = ($refCounts.Values | Measure-Object -Sum).Sum; unique_referenced = $refCounts.Count
        missing_ref_occurrences = $missingRefs.Count; missing_ref_unique = @($missingRefs | Sort-Object -Unique).Count
        orphan_ids = $orphans; reused_items = $reused; unicode_comma_looks = $unicodeCommaLooks
        items_per_look_min = ($itemsPerLook | Measure-Object -Minimum).Minimum
        items_per_look_avg = Mean $itemsPerLook.ToArray(); items_per_look_median = Median $itemsPerLook.ToArray()
        items_per_look_p95 = Percentile $itemsPerLook.ToArray() 0.95; items_per_look_max = ($itemsPerLook | Measure-Object -Maximum).Maximum
        style_regex_match = $styleMatched; style_regex_default = $Looks.Count - $styleMatched
        duplicate_item_ids = $dupItemIds.Count; duplicate_outfit_ids = $dupOutfitIds.Count
        exact_unique_rule_keys = $ruleExact.Count; case_insensitive_collision_groups = $ruleCaseInsensitive.Count
        casual_rules = @($Rules | Where-Object phong_cach -eq "Casual").Count
        generic_body_rules = @($Rules | Where-Object { @($_.dang_nguoi).Count -eq 1 -and $_.dang_nguoi[0] -eq "Mọi vóc dáng" }).Count
        generic_tone_rules = @($Rules | Where-Object { @($_.tone_da).Count -eq 1 -and $_.tone_da[0] -eq "Mọi tone da" }).Count
        noncanonical_body_values = $badBody.Count; body_values = $bodyValues.Count
        noncanonical_tone_values = $badTone.Count; tone_values = $toneValues.Count
        noncanonical_body_top = @($badBody | Group-Object | Sort-Object Count -Descending | ForEach-Object { [pscustomobject]@{ label=$_.Name; count=$_.Count } })
        noncanonical_tone_top = @($badTone | Group-Object | Sort-Object Count -Descending | ForEach-Object { [pscustomobject]@{ label=$_.Name; count=$_.Count } })
        duplicate_reason_groups = $reasonGroups.Count; reason_unique = @($Rules.ly_do_tu_van | Sort-Object -Unique).Count
        reason_min = ($reasonLengths | Measure-Object -Minimum).Minimum; reason_avg = Mean $reasonLengths
        reason_p95 = Percentile $reasonLengths 0.95; reason_max = ($reasonLengths | Measure-Object -Maximum).Maximum
        blank_fields = $blankByField
    }
}

$maleAnalysis = AnalyzePair "Nam" $maleItems $maleLooks $maleRules
$femaleAnalysis = AnalyzePair "Nữ" $femaleItems $femaleLooks $femaleRules

function AnalyzeRuleCollisions([string]$Name, $Items, $Looks) {
    $itemById = MakeOrdinalDict
    foreach ($item in $Items) { $itemById[$item.itemID] = $item }
    $groups = MakeOrdinalDict
    $occurrences = 0

    foreach ($look in $Looks) {
        $ids = @(ParseItemIds $look.items)
        $outfitStyle = ExtractOutfitStyle $look.look
        $season = if ($seasonVi.ContainsKey($look.season)) { $seasonVi[$look.season] } else { $look.season }
        $occasion = if ($occasionVi.ContainsKey($look.occasion)) { $occasionVi[$look.occasion] } else { $look.occasion }
        $context = "$season – $occasion"
        foreach ($id in $ids) {
            if (-not $itemById.ContainsKey($id)) { continue }
            $occurrences++
            $item = $itemById[$id]
            $cat = if ($categoryVi.ContainsKey($item.category.ToLowerInvariant())) { $categoryVi[$item.category.ToLowerInvariant()] } else { $item.category }
            $key = "$cat | $($item.style)"
            if (-not $groups.ContainsKey($key)) {
                $groups[$key] = [pscustomobject]@{
                    key = $key; occurrences = 0; items = MakeOrdinalSet; outlines = MakeOrdinalSet; colors = MakeOrdinalSet
                    contexts = MakeOrdinalSet; outfit_styles = MakeOrdinalSet; first_item = $id; first_outfit = $look.outfitID
                }
            }
            $g = $groups[$key]
            $g.occurrences++
            [void]$g.items.Add($id); [void]$g.outlines.Add([string]$item.outline); [void]$g.colors.Add([string]$item.color)
            [void]$g.contexts.Add($context); [void]$g.outfit_styles.Add($outfitStyle)
        }
    }

    $rows = @($groups.Values | ForEach-Object {
        [pscustomobject]@{
            rule_key = $_.key; occurrences = $_.occurrences; unique_items = $_.items.Count
            outlines = $_.outlines.Count; colors = $_.colors.Count; contexts = $_.contexts.Count
            outfit_styles = $_.outfit_styles.Count; first_item = $_.first_item; first_outfit = $_.first_outfit
        }
    })
    $top = @($rows | Sort-Object @{Expression={($_.unique_items + $_.outlines + $_.colors + $_.contexts + $_.outfit_styles)};Descending=$true}, unique_items -Descending | Select-Object -First 60)
    return [pscustomobject]@{
        name = $Name; source_occurrences = $occurrences; unique_keys = $groups.Count
        keys_multi_item = @($rows | Where-Object unique_items -gt 1).Count
        keys_multi_outline = @($rows | Where-Object outlines -gt 1).Count
        keys_multi_color = @($rows | Where-Object colors -gt 1).Count
        keys_multi_context = @($rows | Where-Object contexts -gt 1).Count
        keys_multi_outfit_style = @($rows | Where-Object outfit_styles -gt 1).Count
        discarded_item_variants = (($rows | ForEach-Object { [math]::Max(0, $_.unique_items - 1) }) | Measure-Object -Sum).Sum
        discarded_outline_variants = (($rows | ForEach-Object { [math]::Max(0, $_.outlines - 1) }) | Measure-Object -Sum).Sum
        discarded_color_variants = (($rows | ForEach-Object { [math]::Max(0, $_.colors - 1) }) | Measure-Object -Sum).Sum
        discarded_context_variants = (($rows | ForEach-Object { [math]::Max(0, $_.contexts - 1) }) | Measure-Object -Sum).Sum
        top = $top
    }
}

$maleCollisions = AnalyzeRuleCollisions "Nam" $maleItems $maleLooks
$femaleCollisions = AnalyzeRuleCollisions "Nữ" $femaleItems $femaleLooks

function EmptyCounts($Rows, [string[]]$Columns) {
    return @($Columns | ForEach-Object {
        $c = $_
        [pscustomobject]@{
            field = $c
            blank = @($Rows | Where-Object { [string]::IsNullOrWhiteSpace([string]$_.$c) }).Count
            distinct = @($Rows | ForEach-Object { $_.$c } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | Sort-Object -Unique).Count
        }
    })
}

$itemColumns = @("itemID", "category", "title", "gender", "style", "outline", "materials", "color", "pattern", "detail", "donning/doffing")
$lookColumns = @("outfitID", "items", "look", "season", "occasion")

$dataDictionary = @(
    [pscustomobject]@{ level="Item"; field="itemID"; meaning="Định danh món đồ trong từng subset"; type="Chuỗi"; layer_b="Dùng để nối look → item, nhưng không được lưu vào JSON đầu ra"; risk="Mất truy vết sau biến đổi" },
    [pscustomobject]@{ level="Item"; field="category"; meaning="8 nhóm món đồ chuẩn hóa"; type="Danh mục"; layer_b="Dịch sang category tiếng Việt và ghép vào rule_key"; risk="Một số nhãn Việt gộp nhiều loại sản phẩm" },
    [pscustomobject]@{ level="Item"; field="title"; meaning="Tên/mô tả ngắn món đồ"; type="Văn bản"; layer_b="Không sử dụng"; risk="Mất thông tin sản phẩm cụ thể" },
    [pscustomobject]@{ level="Item"; field="gender"; meaning="Nhóm giới tính mục tiêu"; type="Danh mục"; layer_b="Không sử dụng trực tiếp; tách file nam/nữ theo subset"; risk="Subset vẫn chứa nhiều item Unisex" },
    [pscustomobject]@{ level="Item"; field="style"; meaning="Phong cách/kiểu món đồ"; type="Văn bản danh mục mở"; layer_b="Ghép vào rule_key; đưa vào prompt Gemini"; risk="Không chuẩn hóa chữ hoa/thường; nhiều giá trị gần trùng" },
    [pscustomobject]@{ level="Item"; field="outline"; meaning="Phom/silhouette"; type="Văn bản danh mục mở"; layer_b="Ánh xạ heuristic sang dang_nguoi"; risk="Coverage thấp; quan hệ phù hợp chưa được kiểm định" },
    [pscustomobject]@{ level="Item"; field="materials"; meaning="Chất liệu"; type="Văn bản danh mục mở"; layer_b="Không sử dụng"; risk="Mất tín hiệu mùa, độ rủ, độ thoải mái" },
    [pscustomobject]@{ level="Item"; field="color"; meaning="Màu món đồ"; type="Văn bản danh mục mở"; layer_b="Ánh xạ heuristic sang tone_da"; risk="Không xét vị trí món đồ, undertone, mắt/tóc, độ sáng/bão hòa" },
    [pscustomobject]@{ level="Item"; field="pattern"; meaning="Họa tiết/cấu trúc bề mặt"; type="Văn bản"; layer_b="Không sử dụng"; risk="Mất tín hiệu tương thích thị giác" },
    [pscustomobject]@{ level="Item"; field="detail"; meaning="Chi tiết thiết kế"; type="Văn bản"; layer_b="Không sử dụng"; risk="Mất thông tin giải thích" },
    [pscustomobject]@{ level="Item"; field="donning/doffing"; meaning="Cách mặc/cởi hoặc khóa"; type="Văn bản"; layer_b="Không sử dụng"; risk="Không ảnh hưởng truy hồi" },
    [pscustomobject]@{ level="Outfit"; field="outfitID"; meaning="Định danh outfit"; type="Chuỗi"; layer_b="Chỉ dùng khi duyệt; không lưu đầu ra"; risk="Không thể truy ngược lý do về outfit gốc" },
    [pscustomobject]@{ level="Outfit"; field="items"; meaning="Danh sách itemID trong outfit"; type="Chuỗi phân tách dấu phẩy"; layer_b="Sinh focal item và category phối cùng"; risk="Chỉ giữ category đối tác, bỏ item/style cụ thể" },
    [pscustomobject]@{ level="Outfit"; field="look"; meaning="Mô tả tổng thể outfit"; type="Văn bản"; layer_b="Regex lấy phong_cach và đưa vào prompt"; risk="Regex thất bại nhiều, gán Casual" },
    [pscustomobject]@{ level="Outfit"; field="season"; meaning="Mùa"; type="6 lớp"; layer_b="Dịch rồi gộp vào boi_canh"; risk="Mất khả năng lọc riêng theo mùa" },
    [pscustomobject]@{ level="Outfit"; field="occasion"; meaning="Dịp"; type="Danh mục"; layer_b="Dịch rồi gộp vào boi_canh"; risk="Mất khả năng lọc riêng theo dịp" }
)

$lineage = @(
    [pscustomobject]@{ output="rule_key"; source="item.category + item.style"; method="Dịch category; nối bằng dấu |"; deterministic="Có"; loss="Không chứa itemID/outfitID; case-sensitive; nhiều bản ghi bị nén" },
    [pscustomobject]@{ output="phong_cach"; source="look.look"; method="Regex đầu chuỗi; thất bại → Casual"; deterministic="Có"; loss="Nam chỉ match 33,3% outfit; nữ 60,2%" },
    [pscustomobject]@{ output="boi_canh"; source="look.season + look.occasion"; method="Dịch và gộp chuỗi"; deterministic="Có"; loss="Mùa và dịp không còn là hai trường lọc độc lập" },
    [pscustomobject]@{ output="dang_nguoi"; source="item.outline"; method="Bảng rule AI đề xuất; exact/substring/default"; deterministic="Có"; loss="Heuristic chưa kiểm định; nhiều nhãn lệch vocabulary hệ thống" },
    [pscustomobject]@{ output="tone_da"; source="item.color"; method="Bảng rule AI đề xuất; exact/substring/default"; deterministic="Có"; loss="Giả định màu món đồ quyết định tone phù hợp; không xét mắt/tóc/undertone" },
    [pscustomobject]@{ output="goi_y_phoi_cung"; source="category của item còn lại trong outfit"; method="Dịch, loại trùng giữ thứ tự"; deterministic="Có"; loss="Bỏ style, màu, chất liệu và item đối tác cụ thể" },
    [pscustomobject]@{ output="ly_do_tu_van"; source="style + partner category + outfit style + season + occasion"; method="Gemini 2.5 Flash Lite, 1 call/rule"; deterministic="Không"; loss="Không log prompt/response/model revision/lỗi; chưa kiểm duyệt" }
)

$auditRows = @(
    [pscustomobject]@{ area="Mục tiêu & phạm vi"; status="Partial"; severity="Major"; evidence="Notebook không nêu research question, tiêu chí thành công hay phạm vi loại trừ Child"; recommendation="Định nghĩa rõ đây là knowledge-base heuristic từ subset English Male/Female, không phải bản gốc đầy đủ" },
    [pscustomobject]@{ area="Nguồn & giấy phép"; status="Fail"; severity="Major"; evidence="Không lưu URL/commit/version/license trong output"; recommendation="Ghi release/commit, CC BY-NC 4.0, citation và tuyên bố đã biến đổi dữ liệu" },
    [pscustomobject]@{ area="Quản lý bí mật"; status="Fail"; severity="Critical"; evidence="API key từng hard-code trong notebook (đã được người dùng thu hồi)"; recommendation="Đọc từ biến môi trường; xóa output/history chứa secret; secret scanning" },
    [pscustomobject]@{ area="Tính tái lập"; status="Fail"; severity="Critical"; evidence="LLM sinh văn bản không seed/version pin; không lưu raw response hay log"; recommendation="Lưu model ID/revision, prompt version/hash, timestamp, raw response, trạng thái retry và checksum input/output" },
    [pscustomobject]@{ area="Kiểm tra input"; status="Partial"; severity="Major"; evidence="Đọc CSV đúng UTF-8 nhưng không assert schema, ID duy nhất, foreign key, null hoặc delimiter"; recommendation="Thêm validation trước khi gọi API và báo cáo lỗi có cấu trúc" },
    [pscustomobject]@{ area="Thiết kế rule_key"; status="Fail"; severity="Critical"; evidence="category + style làm khóa; cơ chế first-wins làm mất outline/color/context khác"; recommendation="Khóa có source_item_id/source_outfit_id hoặc tách rule instance và normalized concept" },
    [pscustomobject]@{ area="Chuẩn hóa phong cách"; status="Fail"; severity="Major"; evidence="Regex nhận diện thấp; phần còn lại gán Casual mà không cờ confidence"; recommendation="Tách style bằng parser/LLM có schema hoặc giữ raw text + extracted_style + confidence" },
    [pscustomobject]@{ area="Mapping dáng người"; status="Fail"; severity="Critical"; evidence="Bảng do AI đề xuất, không chuyên gia/nguồn/validation; nhiều outline về default"; recommendation="Xem là giả thuyết; expert annotation + inter-rater agreement + user study/ablation" },
    [pscustomobject]@{ area="Mapping tone da"; status="Fail"; severity="Critical"; evidence="Bảng do AI đề xuất; nghiên cứu cho thấy ảnh hưởng phụ thuộc màu mắt/tóc và ngữ cảnh"; recommendation="Không khẳng định phù hợp khoa học; đo CIELAB/undertone và đánh giá người dùng đa dạng" },
    [pscustomobject]@{ area="Controlled vocabulary"; status="Fail"; severity="Critical"; evidence="Nhãn có hậu tố như '(kéo dài chân)' không match chính xác filter Qdrant của app"; recommendation="Tách nhãn chuẩn và explanation; validate enum trước khi index" },
    [pscustomobject]@{ area="LLM output parsing"; status="Partial"; severity="Major"; evidence="response_mime_type=json tốt nhưng lại regex object non-greedy và catch-all exception"; recommendation="Parse JSON trực tiếp, validate schema, log lỗi typed, retry exponential backoff" },
    [pscustomobject]@{ area="Theo dõi lỗi/fallback"; status="Fail"; severity="Major"; evidence="Fallback hòa vào dữ liệu mà không có cờ generated/fallback/error"; recommendation="Thêm generation_status, error_type, attempt_count, reviewed_status" },
    [pscustomobject]@{ area="Checkpoint/resume"; status="Fail"; severity="Major"; evidence="Chỉ ghi file ở cuối; gián đoạn phải gọi lại toàn bộ"; recommendation="Ghi JSONL/checkpoint theo batch; cache theo prompt hash" },
    [pscustomobject]@{ area="Kiểm định chất lượng"; status="Fail"; severity="Critical"; evidence="Không human review, sample audit, inter-rater agreement, test set hoặc retrieval evaluation"; recommendation="Thiết kế annotation guideline, double review, Cohen's kappa và đánh giá retrieval/end-to-end" },
    [pscustomobject]@{ area="Điểm mạnh triển khai"; status="Pass"; severity="Observation"; evidence="Hàm tách rõ; UTF-8; deterministic mapping; partner dedup giữ thứ tự; Gemini chỉ sinh lý do"; recommendation="Giữ cấu trúc module nhưng bổ sung provenance, validation và evaluation" }
)

$evidenceRows = @(
    [pscustomobject]@{ source="Feng et al. (2026), FashionStylist"; kind="Preprint + dataset paper"; grade="B (primary, chưa peer review)"; finding="Dataset có expert annotation ở item/outfit và 3 benchmark task; không công bố trường body shape/skin tone trong schema V1."; implication="Layer B của bạn là dữ liệu dẫn xuất mới, không nên gọi các mapping bổ sung là annotation gốc của FashionStylist."; limitation="Preprint 2026; chưa có phản biện chính thức."; url="https://arxiv.org/abs/2604.09249" },
    [pscustomobject]@{ source="FashionStylist official repository"; kind="Primary artifact/documentation"; grade="A cho schema/release"; finding="V1 có Female 500/2406, Male 300/1390, Child 200/841; có bản Trung/Anh; license CC BY-NC 4.0."; implication="Thư mục local thiếu Child và các CSV tiếng Trung/URL nguồn; báo cáo phải ghi subset và attribution."; limitation="Repository documentation, không phải nghiên cứu thực nghiệm độc lập."; url="https://github.com/recsys-benchmark/FashionStylist" },
    [pscustomobject]@{ source="Hidayati et al. (2018)"; kind="ACM Multimedia paper"; grade="B"; finding="Học tương quan body shape–clothing style từ social big data; tác giả cũng lưu ý body-shape calculators chủ quan."; implication="Ủng hộ cách data-driven/validated hơn là bảng rule AI viết tay; không xác nhận từng mapping trong notebook."; limitation="Dữ liệu celebrity/social và mục tiêu tương quan, không chứng minh quy tắc phổ quát."; url="https://doi.org/10.1145/3240508.3240546" },
    [pscustomobject]@{ source="Sattar, Pons-Moll, & Fritz (2019)"; kind="WACV paper"; grade="B"; finding="Clothing category và body shape có tương quan trong dữ liệu online; mô hình conditional tốt hơn shape annotation thủ công."; implication="Body shape có thể là tín hiệu, nhưng cần học/đánh giá trên dữ liệu; 'preference' không đồng nghĩa 'tôn dáng'."; limitation="181 người dùng nữ, nguồn web; selection bias và không phải stylist validation."; url="https://virtualhumans.mpi-inf.mpg.de/papers/sattar2019WACV/sattar2019WACV.pdf" },
    [pscustomobject]@{ source="Chattaraman & Rudd (2006)"; kind="Peer-reviewed survey"; grade="C"; finding="Body size, body image và body cathexis liên quan đến sở thích thuộc tính thẩm mỹ trang phục."; implication="Khuyến nghị không nên suy ra chỉ từ dáng cơ thể; sở thích và tự nhận thức cũng là biến quan trọng."; limitation="199 nữ sinh đại học; cũ và khó khái quát."; url="https://doi.org/10.1177/0887302X0602400104" },
    [pscustomobject]@{ source="Perrett & Sprengelmeyer (2021)"; kind="Peer-reviewed experiments"; grade="B"; finding="Có xu hướng lựa chọn màu trang phục khác giữa khuôn mặt da sáng và rám; trục CIELAB b* giải thích tốt hơn chia warm/cool đơn giản."; implication="Chỉ hỗ trợ khái quát hẹp; không xác nhận bảng black/white/navy… → 4 tone trong notebook."; limitation="Mẫu UK, chủ yếu White; chỉ hai nhóm complexion; stimulus mô phỏng."; url="https://doi.org/10.1177/20416695211053361" },
    [pscustomobject]@{ source="Perrett (2023)"; kind="Peer-reviewed experiments"; grade="B"; finding="Màu mắt có ảnh hưởng trội hơn màu da trong đánh giá màu trang phục; nhấn mạnh skin tone có thể bị đặt sai trọng tâm."; implication="Mapping chỉ dùng skin tone là thiếu biến và có nguy cơ bias; nên xem eye/hair/contrast và preference."; limitation="Nghiên cứu thẩm mỹ tri giác, không phải hệ thống recommender ngoài đời."; url="https://doi.org/10.1037/aca0000626" },
    [pscustomobject]@{ source="Zhang et al. (2026; online 2025)"; kind="Peer-reviewed experiment"; grade="B"; finding="22 màu sweatshirt ảnh hưởng cảm nhận độ sáng da và sở thích người tiêu dùng trẻ Trung Quốc; lightness/colorfulness/blue tint quan trọng."; implication="Cần xét hue, lightness, chroma và bối cảnh dân số; tên màu thô là không đủ."; limitation="Mục tiêu perception/preference, không cung cấp bảng phù hợp phổ quát cho mọi tone."; url="https://doi.org/10.1111/cote.12828" }
)

$inventoryRows = @(
    [pscustomobject]@{ scope="Local"; subset="Male"; file="label_en.csv"; rows=$maleItems.Count; status="Có"; note="Item-level English" },
    [pscustomobject]@{ scope="Local"; subset="Male"; file="look_en.csv"; rows=$maleLooks.Count; status="Có"; note="Outfit-level English" },
    [pscustomobject]@{ scope="Local"; subset="Female"; file="label_en_Female.csv"; rows=$femaleItems.Count; status="Có (đã đổi tên)"; note="Tên official là Female/label_en.csv" },
    [pscustomobject]@{ scope="Local"; subset="Female"; file="look_en_female.csv"; rows=$femaleLooks.Count; status="Có (đã đổi tên)"; note="Tên official là Female/look_en.csv" },
    [pscustomobject]@{ scope="Official V1"; subset="Child"; file="label_en.csv"; rows=841; status="Thiếu local"; note="Item trẻ em" },
    [pscustomobject]@{ scope="Official V1"; subset="Child"; file="look_en.csv"; rows=200; status="Thiếu local"; note="Outfit trẻ em" },
    [pscustomobject]@{ scope="Official V1"; subset="Male/Female/Child"; file="label(p...).csv + look(b...).csv"; rows="N/A"; status="Thiếu local"; note="Bản tiếng Trung có URL link; tổng 6 file" },
    [pscustomobject]@{ scope="Derived"; subset="Male"; file="Layer_B_Male_Knowledge.json"; rows=$maleRules.Count; status="Có"; note="Rule tri thức dẫn xuất" },
    [pscustomobject]@{ scope="Derived"; subset="Female"; file="Layer_B_Female_Knowledge.json"; rows=$femaleRules.Count; status="Có"; note="Rule tri thức dẫn xuất" }
)

function Table([string[]]$Headers, $Rows, [string[]]$Properties, [string]$Class = "") {
    $sb = [System.Text.StringBuilder]::new()
    [void]$sb.Append("<div class='table-wrap'><table class='$(Html $Class)'><thead><tr>")
    foreach ($head in $Headers) { [void]$sb.Append("<th>$(Html $head)</th>") }
    [void]$sb.Append("</tr></thead><tbody>")
    foreach ($row in $Rows) {
        [void]$sb.Append("<tr>")
        foreach ($prop in $Properties) { [void]$sb.Append("<td>$(Html $row.$prop)</td>") }
        [void]$sb.Append("</tr>")
    }
    [void]$sb.Append("</tbody></table></div>")
    return $sb.ToString()
}

function Bars([string]$Title, $Data, [int]$MaxCount = 0) {
    if (-not $Data) { return "" }
    if ($MaxCount -le 0) { $MaxCount = ($Data | Measure-Object count -Maximum).Maximum }
    $sb = [System.Text.StringBuilder]::new()
    [void]$sb.Append("<div class='chart'><h4>$(Html $Title)</h4>")
    foreach ($d in $Data) {
        $w = if ($MaxCount -gt 0) { [math]::Max(1, [math]::Round(100 * $d.count / $MaxCount, 1)) } else { 0 }
        [void]$sb.Append("<div class='bar-row'><div class='bar-label' title='$(Html $d.label)'>$(Html $d.label)</div><div class='bar-track'><div class='bar-fill' style='width:$w%'></div></div><div class='bar-value'>$(Html $d.count)</div></div>")
    }
    [void]$sb.Append("</div>")
    return $sb.ToString()
}

$statusSummary = @($auditRows | Group-Object status | Sort-Object Name | ForEach-Object { [pscustomobject]@{ label=$_.Name; count=$_.Count } })
$severitySummary = @($auditRows | Group-Object severity | Sort-Object Name | ForEach-Object { [pscustomobject]@{ label=$_.Name; count=$_.Count } })

$rawPayload = [ordered]@{
    "Item Nam" = $maleItems
    "Item Nữ" = $femaleItems
    "Outfit Nam" = $maleLooks
    "Outfit Nữ" = $femaleLooks
    "Layer B Nam" = $maleRules
    "Layer B Nữ" = $femaleRules
}
$rawJson = ($rawPayload | ConvertTo-Json -Depth 12 -Compress).Replace('</', '<\/')

$overviewCounts = @(
    [pscustomobject]@{ label="Item Nam"; count=$maleItems.Count }, [pscustomobject]@{ label="Item Nữ"; count=$femaleItems.Count },
    [pscustomobject]@{ label="Outfit Nam"; count=$maleLooks.Count }, [pscustomobject]@{ label="Outfit Nữ"; count=$femaleLooks.Count },
    [pscustomobject]@{ label="Rule Nam"; count=$maleRules.Count }, [pscustomobject]@{ label="Rule Nữ"; count=$femaleRules.Count }
)

$mappingTable = @(
    [pscustomobject]@{ group="Nam"; dimension="Outline → dáng"; exact=$maleCoverage.body_exact; partial=$maleCoverage.body_partial; default=$maleCoverage.body_default; fallback=Pct $maleCoverage.body_default $maleItems.Count },
    [pscustomobject]@{ group="Nam"; dimension="Color → tone"; exact=$maleCoverage.tone_exact; partial=$maleCoverage.tone_partial; default=$maleCoverage.tone_default; fallback=Pct $maleCoverage.tone_default $maleItems.Count },
    [pscustomobject]@{ group="Nữ"; dimension="Outline → dáng"; exact=$femaleCoverage.body_exact; partial=$femaleCoverage.body_partial; default=$femaleCoverage.body_default; fallback=Pct $femaleCoverage.body_default $femaleItems.Count },
    [pscustomobject]@{ group="Nữ"; dimension="Color → tone"; exact=$femaleCoverage.tone_exact; partial=$femaleCoverage.tone_partial; default=$femaleCoverage.tone_default; fallback=Pct $femaleCoverage.tone_default $femaleItems.Count }
)

$collisionSummary = @(
    [pscustomobject]@{ group="Nam"; keys=$maleCollisions.unique_keys; multi_item=$maleCollisions.keys_multi_item; multi_outline=$maleCollisions.keys_multi_outline; multi_color=$maleCollisions.keys_multi_color; multi_context=$maleCollisions.keys_multi_context; lost_items=$maleCollisions.discarded_item_variants; lost_colors=$maleCollisions.discarded_color_variants },
    [pscustomobject]@{ group="Nữ"; keys=$femaleCollisions.unique_keys; multi_item=$femaleCollisions.keys_multi_item; multi_outline=$femaleCollisions.keys_multi_outline; multi_color=$femaleCollisions.keys_multi_color; multi_context=$femaleCollisions.keys_multi_context; lost_items=$femaleCollisions.discarded_item_variants; lost_colors=$femaleCollisions.discarded_color_variants }
)

$ruleQuality = @(
    [pscustomobject]@{ group="Nam"; rules=$maleRules.Count; casual=$maleAnalysis.casual_rules; casual_pct=Pct $maleAnalysis.casual_rules $maleRules.Count; generic_body=$maleAnalysis.generic_body_rules; generic_body_pct=Pct $maleAnalysis.generic_body_rules $maleRules.Count; generic_tone=$maleAnalysis.generic_tone_rules; generic_tone_pct=Pct $maleAnalysis.generic_tone_rules $maleRules.Count; bad_body=Pct $maleAnalysis.noncanonical_body_values $maleAnalysis.body_values; bad_tone=Pct $maleAnalysis.noncanonical_tone_values $maleAnalysis.tone_values },
    [pscustomobject]@{ group="Nữ"; rules=$femaleRules.Count; casual=$femaleAnalysis.casual_rules; casual_pct=Pct $femaleAnalysis.casual_rules $femaleRules.Count; generic_body=$femaleAnalysis.generic_body_rules; generic_body_pct=Pct $femaleAnalysis.generic_body_rules $femaleRules.Count; generic_tone=$femaleAnalysis.generic_tone_rules; generic_tone_pct=Pct $femaleAnalysis.generic_tone_rules $femaleRules.Count; bad_body=Pct $femaleAnalysis.noncanonical_body_values $femaleAnalysis.body_values; bad_tone=Pct $femaleAnalysis.noncanonical_tone_values $femaleAnalysis.tone_values }
)

$relationshipRows = @(
    [pscustomobject]@{ group="Nam"; refs=$maleAnalysis.reference_count; unique=$maleAnalysis.unique_referenced; missing=$maleAnalysis.missing_ref_occurrences; orphan=$maleAnalysis.orphan_ids.Count; reused=$maleAnalysis.reused_items.Count; min=$maleAnalysis.items_per_look_min; avg=$maleAnalysis.items_per_look_avg.ToString("N2"); median=$maleAnalysis.items_per_look_median; p95=$maleAnalysis.items_per_look_p95; max=$maleAnalysis.items_per_look_max; comma=$maleAnalysis.unicode_comma_looks },
    [pscustomobject]@{ group="Nữ"; refs=$femaleAnalysis.reference_count; unique=$femaleAnalysis.unique_referenced; missing=$femaleAnalysis.missing_ref_occurrences; orphan=$femaleAnalysis.orphan_ids.Count; reused=$femaleAnalysis.reused_items.Count; min=$femaleAnalysis.items_per_look_min; avg=$femaleAnalysis.items_per_look_avg.ToString("N2"); median=$femaleAnalysis.items_per_look_median; p95=$femaleAnalysis.items_per_look_p95; max=$femaleAnalysis.items_per_look_max; comma=$femaleAnalysis.unicode_comma_looks }
)

$sourceUrl = "https://github.com/recsys-benchmark/FashionStylist"
$generated = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"

$html = @"
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audit dữ liệu Layer B – FashionStylist</title>
<style>
:root{--ink:#172033;--muted:#64748b;--line:#d9e2ec;--paper:#f6f8fb;--card:#fff;--navy:#173b57;--teal:#147d77;--cyan:#38a3a5;--amber:#d97706;--red:#b42318;--green:#16794b;--soft:#eef5f5}
*{box-sizing:border-box} body{margin:0;font-family:Inter,"Segoe UI",Arial,sans-serif;color:var(--ink);background:var(--paper);line-height:1.45}
header{background:linear-gradient(120deg,#102f49,#176b72);color:#fff;padding:34px max(24px,calc((100vw - 1440px)/2));box-shadow:0 8px 24px #0f294033}
header h1{margin:0 0 7px;font-size:30px;letter-spacing:-.4px} header p{margin:4px 0;color:#d8f1ef}.eyebrow{text-transform:uppercase;letter-spacing:1.7px;font-weight:700;font-size:12px;color:#9de2dc}
.shell{max-width:1440px;margin:auto;padding:22px}.notice{background:#fff7ed;border:1px solid #fed7aa;border-left:6px solid var(--amber);padding:15px 18px;border-radius:10px;margin-bottom:18px}.notice strong{color:#9a3412}
.tabs{display:flex;gap:7px;overflow:auto;padding:3px 0 13px;position:sticky;top:0;background:var(--paper);z-index:5}.tabs button{white-space:nowrap;border:1px solid var(--line);background:#fff;color:var(--navy);padding:9px 13px;border-radius:999px;font-weight:650;cursor:pointer}.tabs button.active{background:var(--navy);color:#fff;border-color:var(--navy)}
.panel{display:none}.panel.active{display:block}.grid{display:grid;gap:14px}.kpis{grid-template-columns:repeat(auto-fit,minmax(180px,1fr));margin:14px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 3px 10px #0f29400a}.kpi .value{font-size:27px;font-weight:780;color:var(--navy)}.kpi .label{font-size:13px;color:var(--muted);margin-top:3px}.kpi.warn .value{color:var(--amber)}.kpi.bad .value{color:var(--red)}
.two{grid-template-columns:repeat(auto-fit,minmax(410px,1fr))}.three{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:17px;box-shadow:0 3px 10px #0f29400a;margin-bottom:14px}.card h3{margin:0 0 12px;font-size:18px;color:var(--navy)}.card h4{margin:0 0 10px;color:var(--navy)}
h2{font-size:24px;margin:10px 0 6px;color:var(--navy)}.lead{color:var(--muted);max-width:1050px;margin-top:0}.mini{font-size:12px;color:var(--muted)}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:9px}table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}th{background:#e8f1f4;color:#173b57;text-align:left;position:sticky;top:0;z-index:1}th,td{padding:9px 10px;border-bottom:1px solid #e7edf2;vertical-align:top}tbody tr:nth-child(even){background:#fafcfd}tbody tr:hover{background:#eef8f7}
.chart{border:1px solid var(--line);padding:14px;border-radius:10px;background:#fff}.bar-row{display:grid;grid-template-columns:minmax(115px,1.5fr) minmax(120px,3fr) 52px;gap:9px;align-items:center;margin:7px 0}.bar-label{font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.bar-track{height:13px;background:#e8eef3;border-radius:20px;overflow:hidden}.bar-fill{height:100%;background:linear-gradient(90deg,var(--teal),var(--cyan));border-radius:20px}.bar-value{font-variant-numeric:tabular-nums;font-size:12px;text-align:right}
.badge{display:inline-block;padding:3px 8px;border-radius:999px;font-size:11px;font-weight:750}.pass{background:#dcfce7;color:#166534}.partial{background:#fef3c7;color:#92400e}.fail{background:#fee2e2;color:#991b1b}.critical{color:#991b1b;font-weight:750}.major{color:#9a3412;font-weight:700}.obs{color:#166534}
.flow{display:flex;align-items:stretch;gap:8px;overflow:auto;padding:8px 0}.flow .step{min-width:170px;flex:1;padding:13px;border:1px solid var(--line);border-radius:10px;background:#fff}.flow .arrow{align-self:center;color:var(--teal);font-size:22px}.flow b{display:block;color:var(--navy);margin-bottom:4px}
.callout{padding:13px 15px;border-radius:9px;background:#eef8f7;border:1px solid #b9dedb;margin:11px 0}.callout.red{background:#fff1f0;border-color:#fecaca}.callout.blue{background:#eff6ff;border-color:#bfdbfe}.cite a{word-break:break-all;color:#0f6b8b}
.explorer-tools{display:flex;flex-wrap:wrap;gap:9px;margin:10px 0}.explorer-tools select,.explorer-tools input{padding:9px 11px;border:1px solid var(--line);border-radius:8px;background:#fff;min-width:220px}.pager{display:flex;gap:8px;align-items:center;margin-top:10px}.pager button{padding:7px 11px;border:1px solid var(--line);background:#fff;border-radius:7px;cursor:pointer}
code{background:#edf2f7;padding:2px 5px;border-radius:5px}ul.tight{margin:6px 0 6px 20px;padding:0}ul.tight li{margin:4px 0}.source-note{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:10px;margin-top:12px}
@media(max-width:720px){.shell{padding:14px}.two,.three{grid-template-columns:1fr}.bar-row{grid-template-columns:110px 1fr 42px}header{padding:25px 18px}}
@media print{body{background:#fff}.tabs{display:none}.panel{display:block!important;break-before:page}.shell{max-width:none}.card,.kpi{box-shadow:none;break-inside:avoid}.explorer{display:none}a{color:#000}.table-wrap{overflow:visible}th{position:static}}
</style>
</head>
<body>
<header><div class="eyebrow">Khóa luận · Dataset audit · Layer B</div><h1>Trực quan và đánh giá học thuật dữ liệu Layer B</h1><p>FashionStylist English Male/Female → knowledge rules cho chatbot tư vấn phối đồ</p><p class="mini" style="color:#cce8e6">Tạo lúc $generated · Phân tích read-only từ CSV, JSON, notebook và lịch sử Git local</p></header>
<main class="shell">
<div class="notice"><strong>Kết luận ngắn:</strong> Notebook có thể xem là <b>prototype xây knowledge base</b>, nhưng <b>chưa đạt chuẩn quy trình xây dựng dữ liệu học thuật</b>. Điểm chặn lớn nhất là mapping AI chưa kiểm định, mất provenance, first-wins dedup, nhãn không khớp filter hệ thống và không có human/LLM quality audit.</div>
<nav class="tabs" id="tabs">
<button class="active" data-tab="overview">Tổng quan</button><button data-tab="inventory">Phạm vi dữ liệu</button><button data-tab="schema">Schema & lineage</button><button data-tab="items">Item</button><button data-tab="looks">Outfit</button><button data-tab="links">Quan hệ</button><button data-tab="rules">Layer B JSON</button><button data-tab="loss">Mất thông tin</button><button data-tab="notebook">Audit notebook</button><button data-tab="evidence">Nguồn học thuật</button><button data-tab="explorer">Tra cứu toàn bộ</button>
</nav>

<section id="overview" class="panel active">
<h2>1. Tổng quan điều hành</h2><p class="lead">Dashboard tách rõ ba lớp: dữ liệu gốc local, dữ liệu Layer B dẫn xuất, và bằng chứng/giới hạn học thuật. Các con số không suy ra chất lượng tư vấn; chúng mô tả coverage, tính toàn vẹn và khả năng tái lập.</p>
<div class="grid kpis">
<div class="kpi"><div class="value">$($maleItems.Count + $femaleItems.Count)</div><div class="label">Item local (Nam + Nữ)</div></div>
<div class="kpi"><div class="value">$($maleLooks.Count + $femaleLooks.Count)</div><div class="label">Outfit local</div></div>
<div class="kpi"><div class="value">$($maleRules.Count + $femaleRules.Count)</div><div class="label">Rule Layer B hiện tại</div></div>
<div class="kpi warn"><div class="value">1.041</div><div class="label">Bản ghi official Child bị thiếu (841 item + 200 outfit)</div></div>
<div class="kpi bad"><div class="value">$(@($auditRows | Where-Object status -eq 'Fail').Count)</div><div class="label">Tiêu chí audit ở trạng thái Fail</div></div>
<div class="kpi"><div class="value">0</div><div class="label">Foreign-key item bị thiếu trong 800 outfit</div></div>
</div>
<div class="grid two">
$(Bars "Quy mô local và đầu ra" $overviewCounts)
<div class="card"><h3>Phát hiện có ảnh hưởng lớn</h3><ul class="tight">
<li>Official V1 còn có <b>Child</b> và bản tiếng Trung có URL nguồn; local hiện không phải bộ FashionStylist đầy đủ.</li>
<li><code>rule_key = category + style</code> làm nhiều biến thể item, màu, phom và bối cảnh rơi vào một khóa; bản gặp đầu tiên quyết định rule.</li>
<li>Regex phong cách thất bại ở $(Pct $maleAnalysis.style_regex_default $maleLooks.Count) outfit nam và $(Pct $femaleAnalysis.style_regex_default $femaleLooks.Count) outfit nữ; chúng bị đổi thành <code>Casual</code>.</li>
<li>$(Pct $maleAnalysis.noncanonical_body_values $maleAnalysis.body_values) giá trị dáng nam và $(Pct $femaleAnalysis.noncanonical_body_values $femaleAnalysis.body_values) giá trị dáng nữ có hậu tố ngoài vocabulary của app, làm filter Qdrant không match chính xác.</li>
<li>Toàn bộ lý do tư vấn đều khác nhau và không rỗng, nhưng điều đó <b>không chứng minh đúng</b>; không có log hoặc human review để xác nhận.</li>
</ul></div>
</div>
<div class="card"><h3>Luồng xây dựng hiện tại</h3><div class="flow"><div class="step"><b>1. CSV item/look</b>Đọc English Male/Female</div><div class="arrow">→</div><div class="step"><b>2. Join</b><code>look.items</code> → <code>itemID</code></div><div class="arrow">→</div><div class="step"><b>3. Heuristic</b>Category, body, skin, context</div><div class="arrow">→</div><div class="step"><b>4. Gemini</b>Sinh <code>ly_do_tu_van</code></div><div class="arrow">→</div><div class="step"><b>5. Dedup</b>First-wins theo rule_key</div><div class="arrow">→</div><div class="step"><b>6. JSON/Qdrant</b>1.296 rule hiện tại</div></div></div>
</section>

<section id="inventory" class="panel">
<h2>2. Phạm vi và kiểm kê file</h2><p class="lead">Đối chiếu local với cấu trúc official V1. Đây là phần nên đưa vào mục “Nguồn và phạm vi dữ liệu” trong khóa luận.</p>
<div class="callout red"><b>Phạm vi chính xác nên viết:</b> “Nghiên cứu sử dụng hai subset tiếng Anh Female và Male của FashionStylist V1 (3.796 item, 800 outfit), không sử dụng subset Child và các CSV tiếng Trung.”</div>
$(Table @("Phạm vi","Subset","File","Số dòng","Trạng thái","Ghi chú") $inventoryRows @("scope","subset","file","rows","status","note"))
<div class="grid two" style="margin-top:14px"><div class="card"><h3>Lịch sử Git local</h3><ul class="tight"><li>JSON nam/nữ ban đầu ở root, sau đó rename 100% sang <code>Fashion_Stylists/</code> và cuối cùng <code>data/stylists/</code>.</li><li>Không có bằng chứng về một JSON nội dung khác trong Git: các vị trí là bản sao do tái cấu trúc.</li><li><code>data/LayerB/</code> đang untracked, nên CSV và notebook chưa có lịch sử version/commit để tái lập.</li><li>Hai notebook tích hợp Layer B cũ từng tồn tại rồi bị xóa, nhưng không phải file knowledge bổ sung.</li></ul></div><div class="card cite"><h3>Giấy phép và attribution</h3><p>Official repository dùng <b>CC BY-NC 4.0</b>: phải ghi công, dẫn link license và nêu rõ đã biến đổi; không dùng thương mại.</p><p><a href="$sourceUrl">$sourceUrl</a></p><p><a href="https://github.com/recsys-benchmark/FashionStylist/blob/main/LICENSE">https://github.com/recsys-benchmark/FashionStylist/blob/main/LICENSE</a></p></div></div>
</section>

<section id="schema" class="panel">
<h2>3. Data dictionary và lineage CSV → Layer B</h2><p class="lead">Schema gốc chứa mô tả item chi tiết và outfit-level context. Notebook chỉ giữ một phần nhỏ, đồng thời bổ sung hai trường suy diễn chưa có trong FashionStylist.</p>
$(Table @("Cấp","Trường","Ý nghĩa","Kiểu","Cách dùng ở Layer B","Rủi ro") $dataDictionary @("level","field","meaning","type","layer_b","risk"))
<h3 style="margin-top:20px">Lineage của 7 trường đầu ra</h3>
$(Table @("Đầu ra","Nguồn","Phương pháp","Deterministic","Mất mát/giới hạn") $lineage @("output","source","method","deterministic","loss"))
</section>

<section id="items" class="panel">
<h2>4. Hồ sơ item-level</h2><p class="lead">Không có ô trống trong 11 trường item ở cả hai subset. Tuy vậy số lượng giá trị mở rất lớn, đặc biệt ở style, color, detail và outline nữ, làm rule-based mapping khó bao phủ.</p>
<div class="grid two">
$(Bars "Category Nam" (CountValues $maleItems "category" 12))
$(Bars "Category Nữ" (CountValues $femaleItems "category" 12))
$(Bars "Gender trong subset Nam" (CountValues $maleItems "gender" 10))
$(Bars "Gender trong subset Nữ" (CountValues $femaleItems "gender" 10))
$(Bars "Màu phổ biến – Nam (top 15)" (CountValues $maleItems "color" 15))
$(Bars "Màu phổ biến – Nữ (top 15)" (CountValues $femaleItems "color" 15))
$(Bars "Outline phổ biến – Nam (top 15)" (CountValues $maleItems "outline" 15))
$(Bars "Outline phổ biến – Nữ (top 15)" (CountValues $femaleItems "outline" 15))
</div>
<div class="card"><h3>Độ đa dạng và ô trống</h3><div class="grid two"><div>$(Table @("Trường Nam","Blank","Distinct") (EmptyCounts $maleItems $itemColumns) @("field","blank","distinct"))</div><div>$(Table @("Trường Nữ","Blank","Distinct") (EmptyCounts $femaleItems $itemColumns) @("field","blank","distinct"))</div></div></div>
<div class="callout blue"><b>Lưu ý giới tính:</b> subset Nam có 176 item Unisex và 1 item Women's; subset Nữ có 1.070 item Unisex. Vì vậy “subset theo giới” không đồng nghĩa mọi item bên trong mang đúng một nhãn gender.</div>
</section>

<section id="looks" class="panel">
<h2>5. Hồ sơ outfit-level</h2><div class="grid two">
$(Bars "Season – Nam" (CountValues $maleLooks "season" 10))
$(Bars "Season – Nữ" (CountValues $femaleLooks "season" 10))
$(Bars "Occasion – Nam" (CountValues $maleLooks "occasion" 10))
$(Bars "Occasion – Nữ" (CountValues $femaleLooks "occasion" 10))
</div>
<div class="grid two"><div class="card"><h3>Regex lấy phong cách</h3><p>Nam: match <b>$($maleAnalysis.style_regex_match)/$($maleLooks.Count)</b> ($(Pct $maleAnalysis.style_regex_match $maleLooks.Count)); default Casual <b>$($maleAnalysis.style_regex_default)</b>.</p><p>Nữ: match <b>$($femaleAnalysis.style_regex_match)/$($femaleLooks.Count)</b> ($(Pct $femaleAnalysis.style_regex_match $femaleLooks.Count)); default Casual <b>$($femaleAnalysis.style_regex_default)</b>.</p><p class="mini">Regex chỉ nhận mẫu bắt đầu bằng “... style”. Mô tả hợp lệ nhưng không đúng template bị hạ thành Casual.</p></div><div class="card"><h3>Schema quality</h3><div class="grid two"><div>$(Table @("Trường Nam","Blank","Distinct") (EmptyCounts $maleLooks $lookColumns) @("field","blank","distinct"))</div><div>$(Table @("Trường Nữ","Blank","Distinct") (EmptyCounts $femaleLooks $lookColumns) @("field","blank","distinct"))</div></div></div></div>
</section>

<section id="links" class="panel">
<h2>6. Toàn vẹn quan hệ outfit–item</h2><p class="lead">Foreign key sạch: không outfit nào trỏ tới item không tồn tại. Các item mồ côi không được notebook biến thành rule vì không thuộc outfit nào.</p>
$(Table @("Nhóm","Tổng refs","Item được tham chiếu","Refs thiếu","Item mồ côi","Item lặp ở nhiều outfit","Min item/look","Mean","Median","P95","Max","Look dùng dấu phẩy Unicode") $relationshipRows @("group","refs","unique","missing","orphan","reused","min","avg","median","p95","max","comma"))
<div class="grid two" style="margin-top:14px"><div class="card"><h3>Item mồ côi Nam ($($maleAnalysis.orphan_ids.Count))</h3><p>$(Html ($maleAnalysis.orphan_ids -join ', '))</p></div><div class="card"><h3>Item mồ côi Nữ ($($femaleAnalysis.orphan_ids.Count))</h3><p>$(Html ($femaleAnalysis.orphan_ids -join ', '))</p></div></div>
</section>

<section id="rules" class="panel">
<h2>7. Audit JSON Layer B hiện tại</h2><p class="lead">Đầu ra đầy đủ 7 trường và không rỗng, nhưng coverage mặc định cao và vocabulary có giá trị không match filter chính xác của app.</p>
$(Table @("Nhóm","Rules","Phong cách Casual","Casual %","Body default","Body default %","Tone default","Tone default %","Body value ngoài enum","Tone value ngoài enum") $ruleQuality @("group","rules","casual","casual_pct","generic_body","generic_body_pct","generic_tone","generic_tone_pct","bad_body","bad_tone"))
<div class="grid two" style="margin-top:14px">
<div class="card"><h3>Chất lượng văn bản Gemini</h3><p>Nam: $($maleAnalysis.reason_unique)/$($maleRules.Count) lý do unique; độ dài trung bình $($maleAnalysis.reason_avg.ToString('N1')) ký tự (min $($maleAnalysis.reason_min), P95 $($maleAnalysis.reason_p95), max $($maleAnalysis.reason_max)).</p><p>Nữ: $($femaleAnalysis.reason_unique)/$($femaleRules.Count) unique; trung bình $($femaleAnalysis.reason_avg.ToString('N1')) ký tự (min $($femaleAnalysis.reason_min), P95 $($femaleAnalysis.reason_p95), max $($femaleAnalysis.reason_max)).</p><p class="mini">Không có duplicate reason hoặc chuỗi rỗng. Đây chỉ là kiểm tra hình thức; không đánh giá factuality, bias hay stylist quality.</p></div>
<div class="card"><h3>Rule-key normalization</h3><p>Exact unique: Nam $($maleAnalysis.exact_unique_rule_keys)/$($maleRules.Count), Nữ $($femaleAnalysis.exact_unique_rule_keys)/$($femaleRules.Count).</p><p>Nữ có <b>$($femaleAnalysis.case_insensitive_collision_groups)</b> nhóm trùng nếu chuẩn hóa chữ hoa/thường: “fresh commuter...” và “Fresh commuter...”. Python dict coi đây là hai khóa khác nhau.</p><p class="mini">Nên normalize Unicode, whitespace và case; đồng thời lưu raw_style riêng.</p></div>
</div>
<div class="grid two">
$(Bars "Body label ngoài enum – Nam" $maleAnalysis.noncanonical_body_top)
$(Bars "Body label ngoài enum – Nữ" $femaleAnalysis.noncanonical_body_top)
$(Bars "Tone label ngoài enum – Nam" $maleAnalysis.noncanonical_tone_top)
$(Bars "Tone label ngoài enum – Nữ" $femaleAnalysis.noncanonical_tone_top)
</div>
<div class="callout red"><b>Bug tích hợp có tác động thật:</b> profile từ vision chỉ trả nhãn chuẩn như <code>Dáng quả lê</code>, nhưng JSON có <code>Dáng quả lê (che đùi)</code>. Qdrant <code>MatchAny</code> dùng exact match, nên các rule có hậu tố không vượt qua filter như dự định.</div>
</section>

<section id="loss" class="panel">
<h2>8. Nén dữ liệu và mất biến thể do rule_key</h2><p class="lead">Các chỉ số dưới đây dựng lại đúng logic notebook trên toàn bộ focal occurrence. Một khóa có nhiều màu/phom/bối cảnh nhưng output chỉ giữ bản đầu tiên.</p>
$(Table @("Nhóm","Unique keys","Key >1 item","Key >1 outline","Key >1 color","Key >1 context","Item variants bị nén","Color variants bị nén") $collisionSummary @("group","keys","multi_item","multi_outline","multi_color","multi_context","lost_items","lost_colors"))
<div class="grid two" style="margin-top:14px"><div class="card"><h3>Coverage mapping trên item gốc</h3>$(Table @("Nhóm","Mapping","Exact","Partial","Default","Default %") $mappingTable @("group","dimension","exact","partial","default","fallback"))</div><div class="card"><h3>Diễn giải đúng</h3><ul class="tight"><li><b>Exact/Partial</b> chỉ có nghĩa chuỗi khớp bảng rule; không có nghĩa khuyến nghị đúng.</li><li><b>Default</b> làm rule trở thành wildcard, tăng recall nhưng giảm cá nhân hóa.</li><li><b>First-wins</b> khiến thứ tự CSV quyết định màu, dáng và context được giữ.</li><li>Compression không được đánh giá bằng held-out retrieval hoặc user study.</li></ul></div></div>
<h3>Top khóa có nhiều biến thể (Nam)</h3>$(Table @("Rule key","Occurrences","Items","Outlines","Colors","Contexts","Outfit styles","First item","First outfit") $maleCollisions.top @("rule_key","occurrences","unique_items","outlines","colors","contexts","outfit_styles","first_item","first_outfit"))
<h3>Top khóa có nhiều biến thể (Nữ)</h3>$(Table @("Rule key","Occurrences","Items","Outlines","Colors","Contexts","Outfit styles","First item","First outfit") $femaleCollisions.top @("rule_key","occurrences","unique_items","outlines","colors","contexts","outfit_styles","first_item","first_outfit"))
</section>

<section id="notebook" class="panel">
<h2>9. Đánh giá notebook theo tiêu chí học thuật</h2><p class="lead">Verdict: <b>Major revision required</b>. Có thể bảo vệ nếu mô tả trung thực là prototype heuristic và bổ sung protocol/validation; chưa nên gọi là bộ quy tắc chuyên gia đã được kiểm chứng.</p>
<div class="grid kpis"><div class="kpi bad"><div class="value">$(@($auditRows | Where-Object status -eq 'Fail').Count)</div><div class="label">Fail</div></div><div class="kpi warn"><div class="value">$(@($auditRows | Where-Object status -eq 'Partial').Count)</div><div class="label">Partial</div></div><div class="kpi"><div class="value">$(@($auditRows | Where-Object status -eq 'Pass').Count)</div><div class="label">Pass</div></div><div class="kpi bad"><div class="value">$(@($auditRows | Where-Object severity -eq 'Critical').Count)</div><div class="label">Critical issues</div></div></div>
<div class="table-wrap"><table><thead><tr><th>Tiêu chí</th><th>Trạng thái</th><th>Mức độ</th><th>Bằng chứng</th><th>Khuyến nghị</th></tr></thead><tbody>
$(($auditRows | ForEach-Object { $sc=$_.status.ToLowerInvariant(); $sev=if($_.severity -eq 'Critical'){'critical'}elseif($_.severity -eq 'Major'){'major'}else{'obs'}; "<tr><td>$(Html $_.area)</td><td><span class='badge $sc'>$(Html $_.status)</span></td><td class='$sev'>$(Html $_.severity)</td><td>$(Html $_.evidence)</td><td>$(Html $_.recommendation)</td></tr>" }) -join "")
</tbody></table></div>
<div class="grid two" style="margin-top:14px"><div class="card"><h3>Cách trình bày có thể bảo vệ</h3><p>“Layer B được xây dựng bằng phép biến đổi dữ liệu thứ cấp có cấu trúc. Các trường category, context và partner category được sinh bằng quy tắc xác định; hai trường body/skin là heuristic do AI đề xuất; trường lý do được sinh bởi Gemini. Do chưa có expert validation, các trường suy diễn được xem là weak labels/prototype knowledge, không phải ground truth chuyên gia.”</p></div><div class="card"><h3>Những câu không nên khẳng định</h3><ul class="tight"><li>“Các mapping phù hợp khoa học cho mọi người.”</li><li>“Layer B là expert-validated knowledge.”</li><li>“1.296 lý do đều chính xác vì không có lỗi JSON.”</li><li>“Sử dụng toàn bộ FashionStylist V1.”</li><li>“Kết quả tái lập hoàn toàn.”</li></ul></div></div>
<div class="card"><h3>Protocol sửa tối thiểu trước bảo vệ</h3><ol><li>Đóng băng nguồn: commit/hash, scope, license, checksum từng CSV.</li><li>Chuẩn hóa schema/enum và lưu <code>source_item_id</code>, <code>source_outfit_id</code>, raw fields.</li><li>Tách label chuẩn khỏi explanation; cấm hậu tố trong filter field.</li><li>Thay first-wins bằng rule instances hoặc aggregation có provenance/conflict fields.</li><li>Version hóa prompt/model; checkpoint JSONL; log status/retry/raw response.</li><li>Lấy mẫu phân tầng để hai người đánh giá; báo Cohen's kappa/percent agreement.</li><li>Đánh giá retrieval trên query set held-out và ablation: không body/tone vs heuristic vs expert-reviewed.</li><li>Ghi AI disclosure và giới hạn bias/cultural generalizability.</li></ol></div>
</section>

<section id="evidence" class="panel">
<h2>10. Bằng chứng chuyên ngành cho mapping</h2><p class="lead">Kết quả literature scan không tìm thấy nguồn đủ mạnh để “hợp thức hóa” nguyên bảng mapping hiện tại. Bằng chứng ủng hộ việc body/appearance có liên quan tới lựa chọn trang phục, nhưng khuyến nghị là đa biến, có tính sở thích/ngữ cảnh và nên được học/kiểm định bằng dữ liệu.</p>
<div class="callout"><b>Kết luận bằng chứng:</b> Giữ mapping hiện tại như <i>heuristic baseline/weak labels</i>, không đổi tên thành “quy tắc chuyên gia”. Nếu muốn nâng thành contribution học thuật, cần expert elicitation + agreement + user evaluation; với tone da nên dùng đặc trưng màu đo được (CIELAB/hue/lightness/chroma) và xem xét mắt, tóc, contrast, văn hóa/sở thích.</div>
<div class="table-wrap cite"><table><thead><tr><th>Nguồn</th><th>Loại</th><th>Grade</th><th>Phát hiện dùng được</th><th>Hàm ý cho Layer B</th><th>Giới hạn</th><th>URL</th></tr></thead><tbody>
$(($evidenceRows | ForEach-Object { "<tr><td>$(Html $_.source)</td><td>$(Html $_.kind)</td><td>$(Html $_.grade)</td><td>$(Html $_.finding)</td><td>$(Html $_.implication)</td><td>$(Html $_.limitation)</td><td><a href='$(Html $_.url)'>$(Html $_.url)</a></td></tr>" }) -join "")
</tbody></table></div>
<div class="card" style="margin-top:14px"><h3>Search strategy có thể tái lập</h3><p><b>Ngày tìm:</b> 2026-07-18. <b>Nguồn:</b> official GitHub/arXiv, ACM/Microsoft Research, WACV author PDF, SAGE/APA/Wiley journal pages, Cornell repository.</p><p><b>Chuỗi tìm chính:</b> “FashionStylist dataset paper”; “body shape clothing silhouette fashion recommendation”; “clothing color skin tone aesthetics”; “eye color skin color clothing aesthetics”.</p><p><b>Inclusion:</b> nguồn primary/official, nghiên cứu thực nghiệm hoặc paper recommender có liên hệ trực tiếp. <b>Exclusion:</b> blog, TikTok/seasonal-color advice không có phương pháp, review AI mơ hồ, nguồn không xác minh được.</p><p><b>Giới hạn:</b> rapid evidence scan, không phải systematic review/PRISMA; văn liệu body/color còn hẹp về giới, chủng tộc và địa lý.</p></div>
</section>

<section id="explorer" class="panel explorer">
<h2>11. Tra cứu toàn bộ bản ghi</h2><p class="lead">Chọn một bảng, tìm trên mọi trường và duyệt theo trang. Dữ liệu được nhúng trực tiếp trong file HTML nên không cần server hay Internet.</p>
<div class="explorer-tools"><select id="datasetSelect"></select><input id="searchBox" placeholder="Tìm mọi trường..."><select id="pageSize"><option>25</option><option selected>50</option><option>100</option></select></div>
<div id="explorerMeta" class="mini"></div><div id="explorerTable" class="table-wrap"></div><div class="pager"><button id="prevPage">← Trước</button><span id="pageInfo"></span><button id="nextPage">Sau →</button></div>
</section>

<div class="source-note">Phân loại bằng chứng và kết luận được áp dụng theo nguyên tắc ARS: tách evidence–inference–recommendation, ưu tiên primary sources, công khai giới hạn và không biến heuristic thành ground truth.</div>
</main>
<script>
const tabs=[...document.querySelectorAll('#tabs button')];
tabs.forEach(b=>b.addEventListener('click',()=>{tabs.forEach(x=>x.classList.remove('active'));document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active');window.scrollTo({top:0,behavior:'smooth'});}));
const datasets=$rawJson;
const ds=document.getElementById('datasetSelect'), search=document.getElementById('searchBox'), size=document.getElementById('pageSize');
let page=1;
Object.keys(datasets).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=`${k} (${datasets[k].length})`;ds.appendChild(o)});
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function renderExplorer(){
 const rows=datasets[ds.value]||[], q=search.value.trim().toLowerCase();
 const filtered=q?rows.filter(r=>Object.values(r).some(v=>String(Array.isArray(v)?v.join(' '):v??'').toLowerCase().includes(q))):rows;
 const n=+size.value, pages=Math.max(1,Math.ceil(filtered.length/n));page=Math.min(page,pages);
 const part=filtered.slice((page-1)*n,page*n), cols=rows.length?Object.keys(rows[0]):[];
 let html='<table><thead><tr>'+cols.map(c=>`<th>${esc(c)}</th>`).join('')+'</tr></thead><tbody>';
 html+=part.map(r=>'<tr>'+cols.map(c=>`<td>${esc(Array.isArray(r[c])?r[c].join(' · '):r[c])}</td>`).join('')+'</tr>').join('')+'</tbody></table>';
 document.getElementById('explorerTable').innerHTML=html;document.getElementById('explorerMeta').textContent=`Hiển thị ${part.length}/${filtered.length} bản ghi khớp (tổng gốc ${rows.length})`;document.getElementById('pageInfo').textContent=`Trang ${page}/${pages}`;
 document.getElementById('prevPage').disabled=page<=1;document.getElementById('nextPage').disabled=page>=pages;
}
ds.addEventListener('change',()=>{page=1;search.value='';renderExplorer()});search.addEventListener('input',()=>{page=1;renderExplorer()});size.addEventListener('change',()=>{page=1;renderExplorer()});document.getElementById('prevPage').addEventListener('click',()=>{page--;renderExplorer()});document.getElementById('nextPage').addEventListener('click',()=>{page++;renderExplorer()});renderExplorer();
</script>
</body></html>
"@

$outDir = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
[System.IO.File]::WriteAllText($OutputPath, $html, [System.Text.UTF8Encoding]::new($false))

$summary = [ordered]@{
    output = $OutputPath
    size_bytes = (Get-Item -LiteralPath $OutputPath).Length
    local_items = $maleItems.Count + $femaleItems.Count
    local_looks = $maleLooks.Count + $femaleLooks.Count
    layer_b_rules = $maleRules.Count + $femaleRules.Count
    official_child_missing_records = 1041
    fail_criteria = @($auditRows | Where-Object status -eq 'Fail').Count
    critical_issues = @($auditRows | Where-Object severity -eq 'Critical').Count
}
$summary | ConvertTo-Json -Depth 5
