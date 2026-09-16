| # | member | decoder | encoder | surgical pre-training | loss | AnatomyLoss | augmentation | input | heads used | ckpts |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `k_dicedet_rules` | Unet++ | ConvNeXt-L (IN-22k→1k, 384) | – | dicedet + f2c 0.25 | GT-statistics | normal | 1024×576 | both | 1 |
| 2 | `k_dicedet_rules_s43` | Unet++ | ConvNeXt-L (IN-22k→1k, 384) | – | dicedet + f2c 0.25 | GT-statistics | normal | 1024×576 | both | 1 |
| 3 | `l_dicedet` | Unet++ | ConvNeXt-L (IN-22k→1k, 384) | – | dicedet + f2c 0.25 | – | normal | 1024×576 | both | 1 |
| 4 | `q_both_dlv3` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | CholecSeg8k+EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 5 | `q_endovis18_dlv3` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 6 | `q_endovis18_dlv3_s43` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 7 | `q_endovis18_dlv3_s44` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 8 | `r_ft_fine` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | – | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 9 | `r_nohflip` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | CholecSeg8k | dicedet + f2c 0.25 | 3D-knowledge | normal, no hflip | 1024×576 | both | 1 |
| 10 | `r_xl` | DeepLabV3+ | ConvNeXt-XL (IN-22k→1k, 384) | CholecSeg8k | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 11 | `s_endovis_s45` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 12 | `s_endovis_s46` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 13 | `s_endovis_s48` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 14 | `s_endovis_s49` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 15 | `s_endovis_s50` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 16 | `s_endovis_s52` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 17 | `s_endovis_s53` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 18 | `s_endovis_s55` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 19 | `s_endovis_s56` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 20 | `s_endovis_s57` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 21 | `s_endovis_s62` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 22 | `s_endovis_s64` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 23 | `s_endovis_s66` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 24 | `s_endovis_s68` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 25 | `s_kdr_seed43` | Unet++ | ConvNeXt-L (IN-22k→1k, 384) | – | dicedet + f2c 0.25 | GT-statistics | normal | 1024×576 | both | 1 |

Total: 25 members, 25 checkpoints
