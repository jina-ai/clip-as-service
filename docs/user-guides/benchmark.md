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


<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-za14{border-color:inherit;text-align:left;vertical-align:bottom}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">model_fullname</th>
    <th class="tg-0pky" rowspan="2">imagenetv2</th>
    <th class="tg-0pky" rowspan="2">voc2007</th>
    <th class="tg-0pky" colspan="7">Class 1</th>
    <th class="tg-0pky" colspan="4">Class 2</th>
    <th class="tg-0pky" colspan="8">Class 3</th>
  </tr>
  <tr>
    <th class="tg-0pky">vtab/caltech101</th>
    <th class="tg-0pky">vtab/cifar10</th>
    <th class="tg-0pky">vtab/cifar100</th>
    <th class="tg-0pky">vtab/dtd</th>
    <th class="tg-0pky">vtab/flowers</th>
    <th class="tg-0pky">vtab/pets</th>
    <th class="tg-0pky">vtab/svhn</th>
    <th class="tg-0pky">vtab/eurosat</th>
    <th class="tg-0pky">vtab/resisc45</th>
    <th class="tg-0pky">vtab/pcam</th>
    <th class="tg-0pky">vtab/diabetic_retinopathy</th>
    <th class="tg-0pky">vtab/clevr_count_all</th>
    <th class="tg-0pky">vtab/clevr_closest_object_distance</th>
    <th class="tg-0pky">vtab/dsprites_label_x_position</th>
    <th class="tg-0pky">vtab/dsprites_label_orientation</th>
    <th class="tg-0pky">vtab/smallnorb_label_azimuth</th>
    <th class="tg-0pky">vtab/smallnorb_label_elevation</th>
    <th class="tg-0pky">vtab/dmlab</th>
    <th class="tg-0pky">vtab/kitti_closest_vehicle_distance</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">RN101 openai</td>
    <td class="tg-0pky">0.561</td>
    <td class="tg-0pky">0.651</td>
    <td class="tg-0pky">0.780</td>
    <td class="tg-0pky">0.807</td>
    <td class="tg-0pky">0.476</td>
    <td class="tg-0pky">0.432</td>
    <td class="tg-0pky">0.652</td>
    <td class="tg-0pky">0.869</td>
    <td class="tg-0pky">0.226</td>
    <td class="tg-0pky">0.314</td>
    <td class="tg-0pky">0.547</td>
    <td class="tg-0pky">0.583</td>
    <td class="tg-0pky">0.280</td>
    <td class="tg-za14">0.242</td>
    <td class="tg-0pky">0.130</td>
    <td class="tg-0pky">0.031</td>
    <td class="tg-0pky">0.021</td>
    <td class="tg-0pky">0.054</td>
    <td class="tg-0pky">0.111</td>
    <td class="tg-0pky">0.139</td>
    <td class="tg-0pky">0.263</td>
  </tr>
  <tr>
    <td class="tg-0pky">RN101 yfcc15m</td>
    <td class="tg-0pky">0.221</td>
    <td class="tg-0pky">0.243</td>
    <td class="tg-0pky">0.469</td>
    <td class="tg-0pky">0.299</td>
    <td class="tg-0pky">0.125</td>
    <td class="tg-0pky">0.117</td>
    <td class="tg-0pky">0.210</td>
    <td class="tg-0pky">0.177</td>
    <td class="tg-0pky">0.137</td>
    <td class="tg-0pky">0.151</td>
    <td class="tg-0pky">0.099</td>
    <td class="tg-0pky">0.479</td>
    <td class="tg-0pky">0.584</td>
    <td class="tg-za14">0.109</td>
    <td class="tg-0pky">0.159</td>
    <td class="tg-0pky">0.031</td>
    <td class="tg-0pky">0.019</td>
    <td class="tg-0pky">0.055</td>
    <td class="tg-0pky">0.097</td>
    <td class="tg-0pky">0.153</td>
    <td class="tg-0pky">0.252</td>
  </tr>
  <tr>
    <td class="tg-0pky">RN50 cc12m</td>
    <td class="tg-0pky">0.224</td>
    <td class="tg-0pky">0.438</td>
    <td class="tg-0pky">0.582</td>
    <td class="tg-0pky">0.395</td>
    <td class="tg-0pky">0.178</td>
    <td class="tg-0pky">0.135</td>
    <td class="tg-0pky">0.095</td>
    <td class="tg-0pky">0.331</td>
    <td class="tg-0pky">0.102</td>
    <td class="tg-0pky">0.148</td>
    <td class="tg-0pky">0.117</td>
    <td class="tg-0pky">0.535</td>
    <td class="tg-0pky">0.293</td>
    <td class="tg-za14">0.184</td>
    <td class="tg-0pky">0.222</td>
    <td class="tg-0pky">0.031</td>
    <td class="tg-0pky">0.025</td>
    <td class="tg-0pky">0.047</td>
    <td class="tg-0pky">0.096</td>
    <td class="tg-0pky">0.161</td>
    <td class="tg-0pky">0.155</td>
  </tr>
  <tr>
    <td class="tg-0pky">RN50 openai</td>
    <td class="tg-0pky">0.529</td>
    <td class="tg-0pky">0.650</td>
    <td class="tg-0pky">0.772</td>
    <td class="tg-0pky">0.715</td>
    <td class="tg-0pky">0.403</td>
    <td class="tg-0pky">0.415</td>
    <td class="tg-0pky">0.660</td>
    <td class="tg-0pky">0.857</td>
    <td class="tg-0pky">0.303</td>
    <td class="tg-0pky">0.408</td>
    <td class="tg-0pky">0.453</td>
    <td class="tg-0pky">0.636</td>
    <td class="tg-0pky">0.171</td>
    <td class="tg-za14">0.217</td>
    <td class="tg-0pky">0.148</td>
    <td class="tg-0pky">0.034</td>
    <td class="tg-0pky">0.014</td>
    <td class="tg-0pky">0.056</td>
    <td class="tg-0pky">0.110</td>
    <td class="tg-0pky">0.145</td>
    <td class="tg-0pky">0.170</td>
  </tr>
  <tr>
    <td class="tg-0pky">RN50 yfcc15m</td>
    <td class="tg-0pky">0.214</td>
    <td class="tg-0pky">0.215</td>
    <td class="tg-0pky">0.402</td>
    <td class="tg-0pky">0.291</td>
    <td class="tg-0pky">0.116</td>
    <td class="tg-0pky">0.122</td>
    <td class="tg-0pky">0.167</td>
    <td class="tg-0pky">0.174</td>
    <td class="tg-0pky">0.157</td>
    <td class="tg-0pky">0.172</td>
    <td class="tg-0pky">0.123</td>
    <td class="tg-0pky">0.533</td>
    <td class="tg-0pky">0.358</td>
    <td class="tg-za14">0.151</td>
    <td class="tg-0pky">0.158</td>
    <td class="tg-0pky">0.032</td>
    <td class="tg-0pky">0.024</td>
    <td class="tg-0pky">0.053</td>
    <td class="tg-0pky">0.120</td>
    <td class="tg-0pky">0.160</td>
    <td class="tg-0pky">0.336</td>
  </tr>
  <tr>
    <td class="tg-0pky">RN50x16 openai</td>
    <td class="tg-0pky">0.643</td>
    <td class="tg-0pky">0.680</td>
    <td class="tg-0pky">0.810</td>
    <td class="tg-0pky">0.813</td>
    <td class="tg-0pky">0.522</td>
    <td class="tg-0pky">0.524</td>
    <td class="tg-0pky">0.724</td>
    <td class="tg-0pky">0.898</td>
    <td class="tg-0pky">0.409</td>
    <td class="tg-0pky">0.433</td>
    <td class="tg-0pky">0.589</td>
    <td class="tg-0pky">0.625</td>
    <td class="tg-0pky">0.715</td>
    <td class="tg-za14">0.195</td>
    <td class="tg-0pky">0.213</td>
    <td class="tg-0pky">0.030</td>
    <td class="tg-0pky">0.026</td>
    <td class="tg-0pky">0.050</td>
    <td class="tg-0pky">0.116</td>
    <td class="tg-0pky">0.146</td>
    <td class="tg-0pky">0.229</td>
  </tr>
  <tr>
    <td class="tg-0pky">RN50x4 openai</td>
    <td class="tg-0pky">0.594</td>
    <td class="tg-0pky">0.682</td>
    <td class="tg-0pky">0.781</td>
    <td class="tg-0pky">0.794</td>
    <td class="tg-0pky">0.451</td>
    <td class="tg-0pky">0.486</td>
    <td class="tg-0pky">0.698</td>
    <td class="tg-0pky">0.887</td>
    <td class="tg-0pky">0.367</td>
    <td class="tg-0pky">0.335</td>
    <td class="tg-0pky">0.532</td>
    <td class="tg-0pky">0.569</td>
    <td class="tg-0pky">0.318</td>
    <td class="tg-za14">0.205</td>
    <td class="tg-0pky">0.082</td>
    <td class="tg-0pky">0.031</td>
    <td class="tg-0pky">0.026</td>
    <td class="tg-0pky">0.056</td>
    <td class="tg-0pky">0.108</td>
    <td class="tg-0pky">0.162</td>
    <td class="tg-0pky">0.233</td>
  </tr>
  <tr>
    <td class="tg-0pky">RN50x64 openai</td>
    <td class="tg-0pky">0.670</td>
    <td class="tg-0pky">0.740</td>
    <td class="tg-0pky">0.834</td>
    <td class="tg-0pky">0.851</td>
    <td class="tg-0pky">0.598</td>
    <td class="tg-0pky">0.531</td>
    <td class="tg-0pky">0.788</td>
    <td class="tg-0pky">0.936</td>
    <td class="tg-0pky">0.481</td>
    <td class="tg-0pky">0.577</td>
    <td class="tg-0pky">0.628</td>
    <td class="tg-0pky">0.539</td>
    <td class="tg-0pky">0.073</td>
    <td class="tg-za14">0.227</td>
    <td class="tg-0pky">0.200</td>
    <td class="tg-0pky">0.034</td>
    <td class="tg-0pky">0.025</td>
    <td class="tg-0pky">0.056</td>
    <td class="tg-0pky">0.125</td>
    <td class="tg-0pky">0.158</td>
    <td class="tg-0pky">0.311</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-B-16 laion400m_e31</td>
    <td class="tg-0pky">0.594</td>
    <td class="tg-0pky">0.767</td>
    <td class="tg-0pky">0.838</td>
    <td class="tg-0pky">0.917</td>
    <td class="tg-0pky">0.712</td>
    <td class="tg-0pky">0.513</td>
    <td class="tg-0pky">0.694</td>
    <td class="tg-0pky">0.892</td>
    <td class="tg-0pky">0.380</td>
    <td class="tg-0pky">0.503</td>
    <td class="tg-0pky">0.585</td>
    <td class="tg-0pky">0.593</td>
    <td class="tg-0pky">0.062</td>
    <td class="tg-za14">0.289</td>
    <td class="tg-0pky">0.245</td>
    <td class="tg-0pky">0.031</td>
    <td class="tg-0pky">0.030</td>
    <td class="tg-0pky">0.059</td>
    <td class="tg-0pky">0.100</td>
    <td class="tg-0pky">0.152</td>
    <td class="tg-0pky">0.200</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-B-16 laion400m_e32</td>
    <td class="tg-0pky">0.597</td>
    <td class="tg-0pky">0.768</td>
    <td class="tg-0pky">0.837</td>
    <td class="tg-0pky">0.917</td>
    <td class="tg-0pky">0.712</td>
    <td class="tg-0pky">0.513</td>
    <td class="tg-0pky">0.692</td>
    <td class="tg-0pky">0.892</td>
    <td class="tg-0pky">0.385</td>
    <td class="tg-0pky">0.501</td>
    <td class="tg-0pky">0.585</td>
    <td class="tg-0pky">0.598</td>
    <td class="tg-0pky">0.077</td>
    <td class="tg-za14">0.287</td>
    <td class="tg-0pky">0.245</td>
    <td class="tg-0pky">0.032</td>
    <td class="tg-0pky">0.029</td>
    <td class="tg-0pky">0.060</td>
    <td class="tg-0pky">0.099</td>
    <td class="tg-0pky">0.151</td>
    <td class="tg-0pky">0.183</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-B-16 openai</td>
    <td class="tg-0pky">0.619</td>
    <td class="tg-0pky">0.783</td>
    <td class="tg-0pky">0.819</td>
    <td class="tg-0pky">0.908</td>
    <td class="tg-0pky">0.669</td>
    <td class="tg-0pky">0.449</td>
    <td class="tg-0pky">0.712</td>
    <td class="tg-0pky">0.890</td>
    <td class="tg-0pky">0.313</td>
    <td class="tg-0pky">0.559</td>
    <td class="tg-0pky">0.582</td>
    <td class="tg-0pky">0.507</td>
    <td class="tg-0pky">0.036</td>
    <td class="tg-za14">0.209</td>
    <td class="tg-0pky">0.158</td>
    <td class="tg-0pky">0.030</td>
    <td class="tg-0pky">0.023</td>
    <td class="tg-0pky">0.053</td>
    <td class="tg-0pky">0.122</td>
    <td class="tg-0pky">0.155</td>
    <td class="tg-0pky">0.263</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-B-16-plus-240 laion400m_e31</td>
    <td class="tg-0pky">0.614</td>
    <td class="tg-0pky">0.764</td>
    <td class="tg-0pky">0.832</td>
    <td class="tg-0pky">0.925</td>
    <td class="tg-0pky">0.733</td>
    <td class="tg-0pky">0.555</td>
    <td class="tg-0pky">0.706</td>
    <td class="tg-0pky">0.904</td>
    <td class="tg-0pky">0.355</td>
    <td class="tg-0pky">0.569</td>
    <td class="tg-0pky">0.615</td>
    <td class="tg-0pky">0.551</td>
    <td class="tg-0pky">0.093</td>
    <td class="tg-za14">0.240</td>
    <td class="tg-0pky">0.159</td>
    <td class="tg-0pky">0.041</td>
    <td class="tg-0pky">0.026</td>
    <td class="tg-0pky">0.056</td>
    <td class="tg-0pky">0.111</td>
    <td class="tg-0pky">0.149</td>
    <td class="tg-0pky">0.280</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-B-16-plus-240 laion400m_e32</td>
    <td class="tg-0pky">0.615</td>
    <td class="tg-0pky">0.764</td>
    <td class="tg-0pky">0.833</td>
    <td class="tg-0pky">0.928</td>
    <td class="tg-0pky">0.738</td>
    <td class="tg-0pky">0.555</td>
    <td class="tg-0pky">0.711</td>
    <td class="tg-0pky">0.902</td>
    <td class="tg-0pky">0.362</td>
    <td class="tg-0pky">0.581</td>
    <td class="tg-0pky">0.613</td>
    <td class="tg-0pky">0.551</td>
    <td class="tg-0pky">0.095</td>
    <td class="tg-za14">0.238</td>
    <td class="tg-0pky">0.160</td>
    <td class="tg-0pky">0.043</td>
    <td class="tg-0pky">0.027</td>
    <td class="tg-0pky">0.054</td>
    <td class="tg-0pky">0.110</td>
    <td class="tg-0pky">0.148</td>
    <td class="tg-0pky">0.281</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-B-32 laion2b_e16</td>
    <td class="tg-0pky">0.573</td>
    <td class="tg-0pky">0.788</td>
    <td class="tg-0pky">0.831</td>
    <td class="tg-0pky">0.941</td>
    <td class="tg-0pky">0.754</td>
    <td class="tg-0pky">0.539</td>
    <td class="tg-0pky">0.691</td>
    <td class="tg-0pky">0.893</td>
    <td class="tg-0pky">0.388</td>
    <td class="tg-0pky">0.503</td>
    <td class="tg-0pky">0.619</td>
    <td class="tg-0pky">0.506</td>
    <td class="tg-0pky">0.195</td>
    <td class="tg-za14">0.192</td>
    <td class="tg-0pky">0.167</td>
    <td class="tg-0pky">0.031</td>
    <td class="tg-0pky">0.024</td>
    <td class="tg-0pky">0.052</td>
    <td class="tg-0pky">0.110</td>
    <td class="tg-0pky">0.189</td>
    <td class="tg-0pky">0.176</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-B-32 laion2b_s34b_b79k</td>
    <td class="tg-0pky">0.581</td>
    <td class="tg-0pky">0.791</td>
    <td class="tg-0pky">0.839</td>
    <td class="tg-0pky">0.936</td>
    <td class="tg-0pky">0.755</td>
    <td class="tg-0pky">0.557</td>
    <td class="tg-0pky">0.716</td>
    <td class="tg-0pky">0.909</td>
    <td class="tg-0pky">0.410</td>
    <td class="tg-0pky">0.482</td>
    <td class="tg-0pky">0.610</td>
    <td class="tg-0pky">0.598</td>
    <td class="tg-0pky">0.734</td>
    <td class="tg-za14">0.153</td>
    <td class="tg-0pky">0.189</td>
    <td class="tg-0pky">0.029</td>
    <td class="tg-0pky">0.034</td>
    <td class="tg-0pky">0.062</td>
    <td class="tg-0pky">0.113</td>
    <td class="tg-0pky">0.159</td>
    <td class="tg-0pky">0.262</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-B-32 laion400m_e31</td>
    <td class="tg-0pky">0.523</td>
    <td class="tg-0pky">0.731</td>
    <td class="tg-0pky">0.818</td>
    <td class="tg-0pky">0.883</td>
    <td class="tg-0pky">0.678</td>
    <td class="tg-0pky">0.521</td>
    <td class="tg-0pky">0.659</td>
    <td class="tg-0pky">0.856</td>
    <td class="tg-0pky">0.220</td>
    <td class="tg-0pky">0.470</td>
    <td class="tg-0pky">0.510</td>
    <td class="tg-0pky">0.549</td>
    <td class="tg-0pky">0.259</td>
    <td class="tg-za14">0.155</td>
    <td class="tg-0pky">0.161</td>
    <td class="tg-0pky">0.033</td>
    <td class="tg-0pky">0.021</td>
    <td class="tg-0pky">0.053</td>
    <td class="tg-0pky">0.117</td>
    <td class="tg-0pky">0.173</td>
    <td class="tg-0pky">0.122</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-B-32 laion400m_e32</td>
    <td class="tg-0pky">0.523</td>
    <td class="tg-0pky">0.733</td>
    <td class="tg-0pky">0.817</td>
    <td class="tg-0pky">0.885</td>
    <td class="tg-0pky">0.677</td>
    <td class="tg-0pky">0.523</td>
    <td class="tg-0pky">0.658</td>
    <td class="tg-0pky">0.854</td>
    <td class="tg-0pky">0.223</td>
    <td class="tg-0pky">0.476</td>
    <td class="tg-0pky">0.510</td>
    <td class="tg-0pky">0.548</td>
    <td class="tg-0pky">0.240</td>
    <td class="tg-za14">0.153</td>
    <td class="tg-0pky">0.161</td>
    <td class="tg-0pky">0.033</td>
    <td class="tg-0pky">0.021</td>
    <td class="tg-0pky">0.054</td>
    <td class="tg-0pky">0.117</td>
    <td class="tg-0pky">0.173</td>
    <td class="tg-0pky">0.118</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-B-32 openai</td>
    <td class="tg-0pky">0.559</td>
    <td class="tg-0pky">0.764</td>
    <td class="tg-0pky">0.815</td>
    <td class="tg-0pky">0.898</td>
    <td class="tg-0pky">0.643</td>
    <td class="tg-0pky">0.443</td>
    <td class="tg-0pky">0.664</td>
    <td class="tg-0pky">0.873</td>
    <td class="tg-0pky">0.135</td>
    <td class="tg-0pky">0.504</td>
    <td class="tg-0pky">0.537</td>
    <td class="tg-0pky">0.623</td>
    <td class="tg-0pky">0.447</td>
    <td class="tg-za14">0.232</td>
    <td class="tg-0pky">0.164</td>
    <td class="tg-0pky">0.037</td>
    <td class="tg-0pky">0.024</td>
    <td class="tg-0pky">0.061</td>
    <td class="tg-0pky">0.127</td>
    <td class="tg-0pky">0.193</td>
    <td class="tg-0pky">0.274</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-g-14 laion2b_s12b_b42k</td>
    <td class="tg-0pky">0.696</td>
    <td class="tg-0pky">0.811</td>
    <td class="tg-0pky">0.851</td>
    <td class="tg-0pky">0.971</td>
    <td class="tg-0pky">0.839</td>
    <td class="tg-0pky">0.682</td>
    <td class="tg-0pky">0.776</td>
    <td class="tg-0pky">0.943</td>
    <td class="tg-0pky">0.603</td>
    <td class="tg-0pky">0.648</td>
    <td class="tg-0pky">0.718</td>
    <td class="tg-0pky">0.560</td>
    <td class="tg-0pky">0.580</td>
    <td class="tg-za14">0.332</td>
    <td class="tg-0pky">0.175</td>
    <td class="tg-0pky">0.036</td>
    <td class="tg-0pky">0.031</td>
    <td class="tg-0pky">0.060</td>
    <td class="tg-0pky">0.115</td>
    <td class="tg-0pky">0.190</td>
    <td class="tg-0pky">0.138</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-H-14 laion2b_s32b_b79k</td>
    <td class="tg-0pky">0.709</td>
    <td class="tg-0pky">0.777</td>
    <td class="tg-0pky">0.850</td>
    <td class="tg-0pky">0.975</td>
    <td class="tg-0pky">0.847</td>
    <td class="tg-0pky">0.678</td>
    <td class="tg-0pky">0.801</td>
    <td class="tg-0pky">0.945</td>
    <td class="tg-0pky">0.563</td>
    <td class="tg-0pky">0.726</td>
    <td class="tg-0pky">0.699</td>
    <td class="tg-0pky">0.542</td>
    <td class="tg-0pky">0.297</td>
    <td class="tg-za14">0.268</td>
    <td class="tg-0pky">0.169</td>
    <td class="tg-0pky">0.032</td>
    <td class="tg-0pky">0.027</td>
    <td class="tg-0pky">0.054</td>
    <td class="tg-0pky">0.111</td>
    <td class="tg-0pky">0.140</td>
    <td class="tg-0pky">0.110</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-L-14 laion2b_s32b_b82k</td>
    <td class="tg-0pky">0.677</td>
    <td class="tg-0pky">0.805</td>
    <td class="tg-0pky">0.851</td>
    <td class="tg-0pky">0.966</td>
    <td class="tg-0pky">0.833</td>
    <td class="tg-0pky">0.629</td>
    <td class="tg-0pky">0.758</td>
    <td class="tg-0pky">0.932</td>
    <td class="tg-0pky">0.459</td>
    <td class="tg-0pky">0.646</td>
    <td class="tg-0pky">0.668</td>
    <td class="tg-0pky">0.563</td>
    <td class="tg-0pky">0.116</td>
    <td class="tg-za14">0.312</td>
    <td class="tg-0pky">0.161</td>
    <td class="tg-0pky">0.032</td>
    <td class="tg-0pky">0.020</td>
    <td class="tg-0pky">0.056</td>
    <td class="tg-0pky">0.108</td>
    <td class="tg-0pky">0.224</td>
    <td class="tg-0pky">0.229</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-L-14 laion400m_e31</td>
    <td class="tg-0pky">0.654</td>
    <td class="tg-0pky">0.758</td>
    <td class="tg-0pky">0.839</td>
    <td class="tg-0pky">0.947</td>
    <td class="tg-0pky">0.774</td>
    <td class="tg-0pky">0.598</td>
    <td class="tg-0pky">0.757</td>
    <td class="tg-0pky">0.917</td>
    <td class="tg-0pky">0.378</td>
    <td class="tg-0pky">0.632</td>
    <td class="tg-0pky">0.671</td>
    <td class="tg-0pky">0.487</td>
    <td class="tg-0pky">0.058</td>
    <td class="tg-za14">0.242</td>
    <td class="tg-0pky">0.149</td>
    <td class="tg-0pky">0.030</td>
    <td class="tg-0pky">0.026</td>
    <td class="tg-0pky">0.053</td>
    <td class="tg-0pky">0.109</td>
    <td class="tg-0pky">0.186</td>
    <td class="tg-0pky">0.200</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-L-14 laion400m_e32</td>
    <td class="tg-0pky">0.654</td>
    <td class="tg-0pky">0.756</td>
    <td class="tg-0pky">0.839</td>
    <td class="tg-0pky">0.946</td>
    <td class="tg-0pky">0.774</td>
    <td class="tg-0pky">0.605</td>
    <td class="tg-0pky">0.756</td>
    <td class="tg-0pky">0.919</td>
    <td class="tg-0pky">0.380</td>
    <td class="tg-0pky">0.622</td>
    <td class="tg-0pky">0.675</td>
    <td class="tg-0pky">0.493</td>
    <td class="tg-0pky">0.061</td>
    <td class="tg-za14">0.243</td>
    <td class="tg-0pky">0.149</td>
    <td class="tg-0pky">0.030</td>
    <td class="tg-0pky">0.026</td>
    <td class="tg-0pky">0.053</td>
    <td class="tg-0pky">0.110</td>
    <td class="tg-0pky">0.186</td>
    <td class="tg-0pky">0.203</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-L-14 openai</td>
    <td class="tg-0pky">0.698</td>
    <td class="tg-0pky">0.783</td>
    <td class="tg-0pky">0.835</td>
    <td class="tg-0pky">0.956</td>
    <td class="tg-0pky">0.758</td>
    <td class="tg-0pky">0.554</td>
    <td class="tg-0pky">0.792</td>
    <td class="tg-0pky">0.932</td>
    <td class="tg-0pky">0.571</td>
    <td class="tg-0pky">0.626</td>
    <td class="tg-0pky">0.633</td>
    <td class="tg-0pky">0.520</td>
    <td class="tg-0pky">0.733</td>
    <td class="tg-za14">0.194</td>
    <td class="tg-0pky">0.161</td>
    <td class="tg-0pky">0.032</td>
    <td class="tg-0pky">0.023</td>
    <td class="tg-0pky">0.045</td>
    <td class="tg-0pky">0.115</td>
    <td class="tg-0pky">0.163</td>
    <td class="tg-0pky">0.218</td>
  </tr>
  <tr>
    <td class="tg-0pky">ViT-L-14-336 openai</td>
    <td class="tg-0pky">0.709</td>
    <td class="tg-0pky">0.781</td>
    <td class="tg-0pky">0.837</td>
    <td class="tg-0pky">0.949</td>
    <td class="tg-0pky">0.744</td>
    <td class="tg-0pky">0.556</td>
    <td class="tg-0pky">0.783</td>
    <td class="tg-0pky">0.937</td>
    <td class="tg-0pky">0.560</td>
    <td class="tg-0pky">0.615</td>
    <td class="tg-0pky">0.638</td>
    <td class="tg-0pky">0.608</td>
    <td class="tg-0pky">0.733</td>
    <td class="tg-za14">0.200</td>
    <td class="tg-0pky">0.158</td>
    <td class="tg-0pky">0.032</td>
    <td class="tg-0pky">0.024</td>
    <td class="tg-0pky">0.046</td>
    <td class="tg-0pky">0.113</td>
    <td class="tg-0pky">0.158</td>
    <td class="tg-0pky">0.262</td>
  </tr>
</tbody>
</table>

## Zero-shot classification

| model_fullname                  | imagenetv2 | voc2007 | vtab/caltech101 | vtab/cifar10 | vtab/cifar100 | vtab/dtd | vtab/flowers | vtab/pets | vtab/svhn | vtab/eurosat | vtab/resisc45 | vtab/pcam | vtab/diabetic_retinopathy | vtab/clevr_count_all | vtab/clevr_closest_object_distance | vtab/dsprites_label_x_position | vtab/dsprites_label_orientation | vtab/smallnorb_label_azimuth | vtab/smallnorb_label_elevation | vtab/dmlab | vtab/kitti_closest_vehicle_distance |
|---------------------------------|------------|---------|-----------------|--------------|---------------|----------|--------------|-----------|-----------|--------------|---------------|-----------|---------------------------|----------------------|------------------------------------|--------------------------------|---------------------------------|------------------------------|--------------------------------|------------|-------------------------------------|
| RN101 openai                    | 0.561      | 0.651   | 0.780           | 0.807        | 0.476         | 0.432    | 0.652        | 0.869     | 0.226     | 0.314        | 0.547         | 0.583     | 0.280                     | 0.242                | 0.130                              | 0.031                          | 0.021                           | 0.054                        | 0.111                          | 0.139      | 0.263                               |
| RN101 yfcc15m                   | 0.221      | 0.243   | 0.469           | 0.299        | 0.125         | 0.117    | 0.210        | 0.177     | 0.137     | 0.151        | 0.099         | 0.479     | 0.584                     | 0.109                | 0.159                              | 0.031                          | 0.019                           | 0.055                        | 0.097                          | 0.153      | 0.252                               |
| RN50 cc12m                      | 0.224      | 0.438   | 0.582           | 0.395        | 0.178         | 0.135    | 0.095        | 0.331     | 0.102     | 0.148        | 0.117         | 0.535     | 0.293                     | 0.184                | 0.222                              | 0.031                          | 0.025                           | 0.047                        | 0.096                          | 0.161      | 0.155                               |
| RN50 openai                     | 0.529      | 0.650   | 0.772           | 0.715        | 0.403         | 0.415    | 0.660        | 0.857     | 0.303     | 0.408        | 0.453         | 0.636     | 0.171                     | 0.217                | 0.148                              | 0.034                          | 0.014                           | 0.056                        | 0.110                          | 0.145      | 0.170                               |
| RN50 yfcc15m                    | 0.214      | 0.215   | 0.402           | 0.291        | 0.116         | 0.122    | 0.167        | 0.174     | 0.157     | 0.172        | 0.123         | 0.533     | 0.358                     | 0.151                | 0.158                              | 0.032                          | 0.024                           | 0.053                        | 0.120                          | 0.160      | 0.336                               |
| RN50x16 openai                  | 0.643      | 0.680   | 0.810           | 0.813        | 0.522         | 0.524    | 0.724        | 0.898     | 0.409     | 0.433        | 0.589         | 0.625     | 0.715                     | 0.195                | 0.213                              | 0.030                          | 0.026                           | 0.050                        | 0.116                          | 0.146      | 0.229                               |
| RN50x4 openai                   | 0.594      | 0.682   | 0.781           | 0.794        | 0.451         | 0.486    | 0.698        | 0.887     | 0.367     | 0.335        | 0.532         | 0.569     | 0.318                     | 0.205                | 0.082                              | 0.031                          | 0.026                           | 0.056                        | 0.108                          | 0.162      | 0.233                               |
| RN50x64 openai                  | 0.670      | 0.740   | 0.834           | 0.851        | 0.598         | 0.531    | 0.788        | 0.936     | 0.481     | 0.577        | 0.628         | 0.539     | 0.073                     | 0.227                | 0.200                              | 0.034                          | 0.025                           | 0.056                        | 0.125                          | 0.158      | 0.311                               |
| ViT-B-16 laion400m_e31          | 0.594      | 0.767   | 0.838           | 0.917        | 0.712         | 0.513    | 0.694        | 0.892     | 0.380     | 0.503        | 0.585         | 0.593     | 0.062                     | 0.289                | 0.245                              | 0.031                          | 0.030                           | 0.059                        | 0.100                          | 0.152      | 0.200                               |
| ViT-B-16 laion400m_e32          | 0.597      | 0.768   | 0.837           | 0.917        | 0.712         | 0.513    | 0.692        | 0.892     | 0.385     | 0.501        | 0.585         | 0.598     | 0.077                     | 0.287                | 0.245                              | 0.032                          | 0.029                           | 0.060                        | 0.099                          | 0.151      | 0.183                               |
| ViT-B-16 openai                 | 0.619      | 0.783   | 0.819           | 0.908        | 0.669         | 0.449    | 0.712        | 0.890     | 0.313     | 0.559        | 0.582         | 0.507     | 0.036                     | 0.209                | 0.158                              | 0.030                          | 0.023                           | 0.053                        | 0.122                          | 0.155      | 0.263                               |
| ViT-B-16-plus-240 laion400m_e31 | 0.614      | 0.764   | 0.832           | 0.925        | 0.733         | 0.555    | 0.706        | 0.904     | 0.355     | 0.569        | 0.615         | 0.551     | 0.093                     | 0.240                | 0.159                              | 0.041                          | 0.026                           | 0.056                        | 0.111                          | 0.149      | 0.280                               |
| ViT-B-16-plus-240 laion400m_e32 | 0.615      | 0.764   | 0.833           | 0.928        | 0.738         | 0.555    | 0.711        | 0.902     | 0.362     | 0.581        | 0.613         | 0.551     | 0.095                     | 0.238                | 0.160                              | 0.043                          | 0.027                           | 0.054                        | 0.110                          | 0.148      | 0.281                               |
| ViT-B-32 laion2b_e16            | 0.573      | 0.788   | 0.831           | 0.941        | 0.754         | 0.539    | 0.691        | 0.893     | 0.388     | 0.503        | 0.619         | 0.506     | 0.195                     | 0.192                | 0.167                              | 0.031                          | 0.024                           | 0.052                        | 0.110                          | 0.189      | 0.176                               |
| ViT-B-32 laion2b_s34b_b79k      | 0.581      | 0.791   | 0.839           | 0.936        | 0.755         | 0.557    | 0.716        | 0.909     | 0.410     | 0.482        | 0.610         | 0.598     | 0.734                     | 0.153                | 0.189                              | 0.029                          | 0.034                           | 0.062                        | 0.113                          | 0.159      | 0.262                               |
| ViT-B-32 laion400m_e31          | 0.523      | 0.731   | 0.818           | 0.883        | 0.678         | 0.521    | 0.659        | 0.856     | 0.220     | 0.470        | 0.510         | 0.549     | 0.259                     | 0.155                | 0.161                              | 0.033                          | 0.021                           | 0.053                        | 0.117                          | 0.173      | 0.122                               |
| ViT-B-32 laion400m_e32          | 0.523      | 0.733   | 0.817           | 0.885        | 0.677         | 0.523    | 0.658        | 0.854     | 0.223     | 0.476        | 0.510         | 0.548     | 0.240                     | 0.153                | 0.161                              | 0.033                          | 0.021                           | 0.054                        | 0.117                          | 0.173      | 0.118                               |
| ViT-B-32 openai                 | 0.559      | 0.764   | 0.815           | 0.898        | 0.643         | 0.443    | 0.664        | 0.873     | 0.135     | 0.504        | 0.537         | 0.623     | 0.447                     | 0.232                | 0.164                              | 0.037                          | 0.024                           | 0.061                        | 0.127                          | 0.193      | 0.274                               |
| ViT-g-14 laion2b_s12b_b42k      | 0.696      | 0.811   | 0.851           | 0.971        | 0.839         | 0.682    | 0.776        | 0.943     | 0.603     | 0.648        | 0.718         | 0.560     | 0.580                     | 0.332                | 0.175                              | 0.036                          | 0.031                           | 0.060                        | 0.115                          | 0.190      | 0.138                               |
| ViT-H-14 laion2b_s32b_b79k      | 0.709      | 0.777   | 0.850           | 0.975        | 0.847         | 0.678    | 0.801        | 0.945     | 0.563     | 0.726        | 0.699         | 0.542     | 0.297                     | 0.268                | 0.169                              | 0.032                          | 0.027                           | 0.054                        | 0.111                          | 0.140      | 0.110                               |
| ViT-L-14 laion2b_s32b_b82k      | 0.677      | 0.805   | 0.851           | 0.966        | 0.833         | 0.629    | 0.758        | 0.932     | 0.459     | 0.646        | 0.668         | 0.563     | 0.116                     | 0.312                | 0.161                              | 0.032                          | 0.020                           | 0.056                        | 0.108                          | 0.224      | 0.229                               |
| ViT-L-14 laion400m_e31          | 0.654      | 0.758   | 0.839           | 0.947        | 0.774         | 0.598    | 0.757        | 0.917     | 0.378     | 0.632        | 0.671         | 0.487     | 0.058                     | 0.242                | 0.149                              | 0.030                          | 0.026                           | 0.053                        | 0.109                          | 0.186      | 0.200                               |
| ViT-L-14 laion400m_e32          | 0.654      | 0.756   | 0.839           | 0.946        | 0.774         | 0.605    | 0.756        | 0.919     | 0.380     | 0.622        | 0.675         | 0.493     | 0.061                     | 0.243                | 0.149                              | 0.030                          | 0.026                           | 0.053                        | 0.110                          | 0.186      | 0.203                               |
| ViT-L-14 openai                 | 0.698      | 0.783   | 0.835           | 0.956        | 0.758         | 0.554    | 0.792        | 0.932     | 0.571     | 0.626        | 0.633         | 0.520     | 0.733                     | 0.194                | 0.161                              | 0.032                          | 0.023                           | 0.045                        | 0.115                          | 0.163      | 0.218                               |
| ViT-L-14-336 openai             | 0.709      | 0.781   | 0.837           | 0.949        | 0.744         | 0.556    | 0.783        | 0.937     | 0.560     | 0.615        | 0.638         | 0.608     | 0.733                     | 0.200                | 0.158                              | 0.032                          | 0.024                           | 0.046                        | 0.113                          | 0.158      | 0.262                               |