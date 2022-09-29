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
| ViT-B-32::laion2b-s34B-b79k           | 577             | 2.94                | 1.40                 |
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
| ViT-H-14::laion2b-s32B-b79k           | 3762            | 4.45                | 3.26                 |
| ViT-g-14::laion2b-s12B-b42k           | 5214            | 5.16                | 4.00                 |
| M-CLIP/LABSE-Vit-L-14                 | 3609            | 4.30                | 4.70                 |
| M-CLIP/XLM-Roberta-Large-Vit-B-32     | 4284            | 5.37                | 1.68                 |
| M-CLIP/XLM-Roberta-Large-Vit-B-16Plus | 4293            | 4.30                | 4.13                 |
| M-CLIP/XLM-Roberta-Large-Vit-L-14     | 4293            | 4.30                | 4.97                 |
 

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