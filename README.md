# icra2023_synthetic_data_pretraining_for_robotics
<img src="https://user-images.githubusercontent.com/23158313/234160232-b9556e11-6797-4233-a4e2-d5eb9d279e8f.png" alt="drawing" width="800"/>

## Requirements

- Install [Pytorch](https://pytorch.org/get-started/locally/)

Example using Windows + CUDA11.7 + pip:
![Pytorch](https://user-images.githubusercontent.com/23158313/233946613-4ae9414a-112e-4be4-a111-131209d40ee6.png)
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```


- Install dependencies
```shell
pip install -r requirements.txt
```

## Clone this repository

```shell
git clone https://github.com/AISciencePlatform/icra2023_synthetic_data_pretraining_for_robotics.git --recursive
```
- Download the datasets from
 
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7860758.svg)](https://doi.org/10.5281/zenodo.7860758)

- Unzip the file into the folder `icra2023_synthetic_data_pretraining_for_robotics`
 
Be sure to have the following structure

<img src="https://user-images.githubusercontent.com/23158313/234159208-14b46d09-0523-4f38-ab62-711bdcea96c3.png" alt="drawing" width="400"/>

## Predictions using pre-trained models

- To test the predictions using the pre-trained models run the `main.py` script.
