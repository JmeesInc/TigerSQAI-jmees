| # | member | decoder | encoder | surgical pre-training | loss | AnatomyLoss | augmentation | input | heads used | ckpts |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `l_dicedet` | Unet++ | ConvNeXt-L (IN-22k→1k, 384) | – | dicedet + f2c 0.25 | – | normal | 1024×576 | both | 1 |
| 2 | `q_both_dlv3` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | CholecSeg8k+EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 3 | `q_endovis18_dlv3` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | EndoVis18 | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 4 | `r_detbd` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | CholecSeg8k | dicedetboundary + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 5 | `r_ft_fine` | DeepLabV3+ | ConvNeXt-L (IN-22k→1k, 384) | – | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 6 | `r_xl` | DeepLabV3+ | ConvNeXt-XL (IN-22k→1k, 384) | CholecSeg8k | dicedet + f2c 0.25 | 3D-knowledge | normal | 1024×576 | both | 1 |
| 7 | `s_kdr_seed43` | Unet++ | ConvNeXt-L (IN-22k→1k, 384) | – | dicedet + f2c 0.25 | GT-statistics | normal | 1024×576 | both | 1 |

Total: 7 members, 7 checkpoints
