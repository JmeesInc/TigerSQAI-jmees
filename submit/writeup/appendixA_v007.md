| # | member | decoder | encoder | surgical pre-training | loss | AnatomyLoss | augmentation | input | heads used | ckpts |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `e_convnext_xl_384` | Unet++ | ConvNeXt-XL (IN-22k→1k, 384) | – | dice + f2c 0.25 | – | normal | 1024×576 | both | 1 |
| 2 | `ft_t2_fine` | Unet++ | ConvNeXt-L (IN-22k→1k, 384) | – | dicedet + f2c 0.25 | GT-statistics | normal | 1024×576 | fine only, w=4 | 1 |
| 3 | `k_dicedet_rules` | Unet++ | ConvNeXt-L (IN-22k→1k, 384) | – | dicedet + f2c 0.25 | GT-statistics | normal | 1024×576 | both | 1 |
| 4 | `q_both_dlv3` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | CholecSeg8k+EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 5 | `q_cholec_dlv3` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | CholecSeg8k | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 6 | `q_cholec_dlv3_s43` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | CholecSeg8k | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 7 | `q_cholec_dlv3_s44` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | CholecSeg8k | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 8 | `q_endovis18_dlv3` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 9 | `q_endovis18_dlv3_s43` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 10 | `q_endovis18_dlv3_s44` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 11 | `q_endovis18_upp` | Unet++ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 12 | `r_nohflip` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | CholecSeg8k | dicedet + f2c 0.25 | 3D-knowledge | normal, no hflip | 1024×576 | both | 1 |
| 13 | `r_toolpaste` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | CholecSeg8k | dicedet + f2c 0.25 | 3D-knowledge | normal, ToolPaste | 1024×576 | both | 1 |
| 14 | `u_sc512_ev_s43` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 896×512 | both | 1 |
| 15 | `u_sc544_ch` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | CholecSeg8k | dicedet + f2c 0.25 | 3D-knowledge | normal | 960×544 | both | 1 |
| 16 | `u_sc544_ev` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 960×544 | both | 1 |

Total: 16 members, 16 checkpoints
