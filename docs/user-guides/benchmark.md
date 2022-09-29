# CLIP Benchmark

In order to evaluate the performance of different CLIP models, we conducted a benchmark on a series of tasks using different datasets. You can find the benchmark results in the following tables. The best results are highlighted in bold. They can be used as a guide to choose the best model for your application.


## Basic statistics

We include the disk usage (in delta) and the peak RAM and VRAM usage (in delta) when running on a single Nvidia TITAN RTX GPU (24GB VRAM) for a series of text and image encoding tasks with `batch_size=8` using PyTorch runtime.

| Model                                 | Disk Usage (MB) | Peak RAM Usage (GB) | Peak VRAM Usage (GB) |
|---------------------------------------|-----------------|---------------------|----------------------|
| RN50::openai                          | **244**         | 2.99                | **1.36**             |
| RN50::yfcc15m                         | 389             | 2.86                | **1.36**             |
| RN50::cc12m                           | 389             | **2.84**            | **1.36**             |
| RN101::openai                         | 278             | 3.05                | 1.40                 |
| RN101::yfcc15m                        | 457             | 2.88                | 1.40                 |
| RN50x4::openai                        | 402             | 3.23                | 1.63                 |
| RN50x16::openai                       | 631             | 3.63                | 2.02                 |
| RN50x64::openai                       | 1291            | 4.08                | 2.98                 |
| ViT-B-32::openai                      | 338             | 3.20                | 1.40                 |
| ViT-B-32::laion400m_e31               | 577             | 2.93                | 1.40                 |
| ViT-B-32::laion400m_e32               | 577             | 2.94                | 1.40                 |
| ViT-B-32::laion2b_e16                 | 577             | 2.93                | 1.40                 |
| ViT-B-32::laion2b-s34b-b79k           | 577             | 2.94                | 1.40                 |
| ViT-B-16::openai                      | 335             | 3.20                | 1.44                 |
| ViT-B-16::laion400m_e31               | 571             | 2.93                | 1.44                 |
| ViT-B-16::laion400m_e32               | 571             | 2.94                | 1.44                 |
| ViT-B-16-plus-240::laion400m_e31      | 795             | 3.03                | 1.59                 |
| ViT-B-16-plus-240::laion400m_e32      | 795             | 3.03                | 1.59                 |
| ViT-L-14::openai                      | 890             | 3.66                | 2.04                 |
| ViT-L-14::laion400m_e31               | 1631            | 3.43                | 2.03                 |
| ViT-L-14::laion400m_e32               | 1631            | 3.42                | 2.03                 |
| ViT-L-14::laion2b-s32b-b82k           | 1631            | 3.43                | 2.03                 |
| ViT-L-14-336::openai                  | 891             | 3.74                | 2.23                 |
| ViT-H-14::laion2b-s32b-b79k           | 3762            | 4.45                | 3.26                 |
| ViT-g-14::laion2b-s12b-b42k           | 5214            | 5.16                | 4.00                 |
| M-CLIP/LABSE-Vit-L-14                 | 3609            | 4.30                | 4.70                 |
| M-CLIP/XLM-Roberta-Large-Vit-B-32     | 4284            | 5.37                | 1.68                 |
| M-CLIP/XLM-Roberta-Large-Vit-B-16Plus | 4293            | 4.30                | 4.13                 |
| M-CLIP/XLM-Roberta-Large-Vit-L-14     | 4293            | 4.30                | 4.97                 |
 

## Zero-shot retrieval

| Model                            | COCO Caption |       | Flickr 8k |       | Flickr 30k |       |
|----------------------------------|--------------|-------|-----------|-------|------------|-------|
|                                  | Image        | Text  | Image     | Text  | Image      | Text  |
| RN101::openai                    | 0.555        | 0.745 | 0.523     | 0.694 | 0.415      | 0.629 |
| RN101::yfcc15m                   | 0.376        | 0.549 | 0.251     | 0.417 | 0.156      | 0.296 |
| RN50::cc12m                      | 0.446        | 0.607 | 0.302     | 0.435 | 0.204      | 0.316 |
| RN50::openai                     | 0.529        | 0.728 | 0.504     | 0.690 | 0.392      | 0.621 |
| RN50::yfcc15m                    | 0.361        | 0.534 | 0.238     | 0.394 | 0.146      | 0.278 |
| RN50x16::openai                  | 0.600        | 0.787 | 0.597     | 0.768 | 0.496      | 0.713 |
| RN50x4::openai                   | 0.581        | 0.767 | 0.558     | 0.729 | 0.451      | 0.671 |
| RN50x64::openai                  | 0.599        | 0.803 | 0.629     | 0.790 | 0.534      | 0.756 |
| ViT-B-16::laion400m_e31          | 0.637        | 0.796 | 0.620     | 0.765 | 0.506      | 0.697 |
| ViT-B-16::laion400m_e32          | 0.636        | 0.796 | 0.620     | 0.767 | 0.508      | 0.697 |
| ViT-B-16::openai                 | 0.584        | 0.767 | 0.564     | 0.727 | 0.452      | 0.671 |
| ViT-B-16-plus-240::laion400m_e31 | 0.660        | 0.809 | 0.642     | 0.788 | 0.533      | 0.725 |
| ViT-B-16-plus-240::laion400m_e32 | 0.662        | 0.811 | 0.644     | 0.791 | 0.535      | 0.727 |
| ViT-B-32::laion2b_e16            | 0.647        | 0.795 | 0.622     | 0.760 | 0.507      | 0.687 |
| ViT-B-32::laion2b_s34b_b79k      | 0.654        | 0.798 | 0.629     | 0.778 | 0.513      | 0.694 |
| ViT-B-32::laion400m_e31          | 0.600        | 0.763 | 0.562     | 0.736 | 0.438      | 0.633 |
| ViT-B-32::laion400m_e32          | 0.600        | 0.765 | 0.562     | 0.736 | 0.437      | 0.634 |
| ViT-B-32::openai                 | 0.560        | 0.749 | 0.532     | 0.699 | 0.413      | 0.629 |
| ViT-g-14::laion2b_s12b_b42k      | 0.724        | 0.853 | 0.730     | 0.846 | 0.639      | 0.806 |
| ViT-H-14::laion2b_s32b_b79k      | 0.734        | 0.861 | 0.746     | 0.856 | 0.657      | 0.823 |
| ViT-L-14::laion2b_s32b_b82k      | 0.711        | 0.840 | 0.712     | 0.824 | 0.620      | 0.789 |
| ViT-L-14::laion400m_e31          | 0.680        | 0.821 | 0.675     | 0.806 | 0.570      | 0.751 |
| ViT-L-14::laion400m_e32          | 0.680        | 0.821 | 0.675     | 0.806 | 0.570      | 0.751 |
| ViT-L-14::openai                 | 0.610        | 0.793 | 0.599     | 0.767 | 0.494      | 0.717 |
| ViT-L-14-336::openai             | 0.616        | 0.812 | 0.629     | 0.779 | 0.533      | 0.741 |

## Zero-shot classification


````{dropdown} Zero-shot retrieval: MS COCO Captions

| model_fullname                   | image_retrieval_recall@5 | text_retrieval_recall@5 |
|----------------------------------|--------------------------|-------------------------|
| RN50::openai                     | 0.5291883349             | 0.7282000184            |
| RN50::yfcc15m                    | 0.3610555828             | 0.5338000059            |
| RN50::cc12m                      | 0.4464214444             | 0.6065999866            |
| RN101::openai                    | 0.5550180078             | 0.7447999716            |
| RN101::yfcc15m                   | 0.3760095835             | 0.5490000248            |
| RN50x4::openai                   | 0.5814074278             | 0.7670000196            |
| RN50x16::openai                  | 0.6001599431             | 0.7868000269            |
| RN50x64::openai                  | 0.5992003083             | 0.8033999801            |
| ViT-B-32::openai                 | 0.5596161485             | 0.7491999865            |
| ViT-B-32::laion400m_e31          | 0.600039959              | 0.7630000114            |
| ViT-B-32::laion400m_e32          | 0.6000000238             | 0.7645999789            |
| ViT-B-32::laion2b_e16            | 0.6468212605             | 0.7950000167            |
| ViT-B-32::laion2b_s34b_b79k      | 0.6540184021             | 0.7983999848            |
| ViT-B-16::openai                 | 0.5842063427             | 0.7671999931            |
| ViT-B-16::laion400m_e31          | 0.6368252635             | 0.7961999774            |
| ViT-B-16::laion400m_e32          | 0.6363854408             | 0.7964000106            |
| ViT-B-16-plus-240::laion400m_e31 | 0.6604158282             | 0.8090000153            |
| ViT-B-16-plus-240::laion400m_e32 | 0.6618952155             | 0.8108000159            |
| ViT-L-14::openai                 | 0.610355854              | 0.793200016             |
| ViT-L-14::laion400m_e31          | 0.679688096              | 0.82099998              |
| ViT-L-14::laion400m_e32          | 0.6801279783             | 0.8212000132            |
| ViT-L-14::laion2b_s32b_b82k      | 0.7109556198             | 0.8399999738            |
| ViT-L-14-336::openai             | 0.6162734628             | 0.8123999834            |
| ViT-H-14::laion2b_s32b_b79k      | 0.7339064479             | 0.8605999947            |
| ViT-g-14::laion2b_s12b_b42k      | 0.7235905528             | 0.853399992             |

````

````{dropdown} Zero-shot classification: ImageNetV2

| model_fullname                   | acc1   | acc5   | mean_per_class_recall |
|----------------------------------|--------|--------|-----------------------|
| RN50::openai                     | 0.5287 | 0.8148 | 0.5291                |
| RN50::yfcc15m                    | 0.2139 | 0.4253 | 0.2145                |
| RN50::cc12m                      | 0.2238 | 0.4563 | 0.2244                |
| RN101::openai                    | 0.5608 | 0.8314 | 0.5617                |
| RN101::yfcc15m                   | 0.2212 | 0.4397 | 0.2216                |
| RN50x4::openai                   | 0.5944 | 0.8584 | 0.5946                |
| RN50x16::openai                  | 0.6427 | 0.8837 | 0.643                 |
| RN50x64::openai                  | 0.6703 | 0.907  | 0.6702                |
| ViT-B-32::openai                 | 0.5594 | 0.8339 | 0.5595                |
| ViT-B-32::laion400m_e31          | 0.5226 | 0.794  | 0.5233                |
| ViT-B-32::laion400m_e32          | 0.5232 | 0.7947 | 0.5235                |
| ViT-B-32::laion2b_e16            | 0.5729 | 0.8391 | 0.5737                |
| ViT-B-32::laion2b_s34b_b79k      | 0.5814 | 0.8392 | 0.5808                |
| ViT-B-16::openai                 | 0.6186 | 0.8735 | 0.6189                |
| ViT-B-16::laion400m_e31          | 0.5942 | 0.8527 | 0.5941                |
| ViT-B-16::laion400m_e32          | 0.5965 | 0.8542 | 0.5963                |
| ViT-B-16-plus-240::laion400m_e31 | 0.6139 | 0.8631 | 0.6146                |
| ViT-B-16-plus-240::laion400m_e32 | 0.6147 | 0.8646 | 0.614                 |
| ViT-L-14::openai                 | 0.6983 | 0.9092 | 0.6986                |
| ViT-L-14::laion400m_e31          | 0.6543 | 0.886  | 0.6547                |
| ViT-L-14::laion400m_e32          | 0.6539 | 0.8857 | 0.6543                |
| ViT-L-14::laion2b_s32b_b82k      | 0.6774 | 0.9024 | 0.6783                |
| ViT-L-14-336::openai             | 0.7094 | 0.9164 | 0.7094                |
| ViT-H-14::laion2b_s32b_b79k      | 0.7087 | 0.9166 | 0.7091                |
| ViT-g-14::laion2b_s12b_b42k      | 0.6956 | 0.9086 | 0.6962                |

````