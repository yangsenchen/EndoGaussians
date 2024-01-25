

# EndoGaussians: Single View Dynamic Gaussian Splatting for Deformable Endoscopic Tissues Reconstruction

Official code for https://arxiv.org/abs/2401.13352


<img src="figures/teaser.png" alt="Reconstructed Image" style="zoom:75%;" />
<!-- <img src="figures/depth0.png" alt="Reconstructed Depth" style="zoom:50%;" /> -->

## Installation
```
git clone https://github.com/yangsenchen/EndoGaussians.git
conda env create --file environment.yml
conda activate endogaussians
pip install -r requirements.txt
```

## Acknowledgement
* This code is developed based on [Dynamic3DGaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians) (Luiten et al.), and [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al.), [EndoNeRF](https://github.com/med-air/EndoNeRF) (Wang et at.).