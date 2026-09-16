# expA19_convnext_variants

**expA17（ConvNeXt-base + Unet++ / 20ep）と完全同一レシピで、encoder の事前学習重みだけを差し替える**
（ユーザー提案 2026-09-08）。ConvNeXt は timm に多様な事前学習が揃っている。

## 比較する重み

| key | timm タグ | 中身 |
|-----|-----------|------|
| （既存 expA17） | `convnext_base.fb_in22k_ft_in1k` | ImageNet-22k -> 1k（timm の既定タグ） |
| in1k | `convnext_base.fb_in1k` | 素の ImageNet-1k |
| clip | `convnext_base.clip_laion2b_augreg_ft_in12k_in1k` | CLIP LAION-2B -> IN-12k -> IN-1k |
| dinov3b | `convnext_base.dinov3_lvd1689m` | DINOv3（LVD-1689M 自己教師あり、ViT-7B から蒸留） |
| dinov3l | `convnext_large.dinov3_lvd1689m` | 同上の large（196M） |

## 注意

- **正規化は全条件で ImageNet mean/std に固定**。CLIP タグは本来 mean/std が異なる
  (0.481/0.269) が、変数を「重みのみ」に絞るため揃えている。CLIP には僅かに不利な条件
- 既存の expA17 fold0 = 0.6636 / expA16(Cholec) fold0 = 0.6602 が比較基準
- まず fold0 ゲート（25 分/本）。勝った重みだけ 5-fold に展開する
