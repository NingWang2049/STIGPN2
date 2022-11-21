Code for paper "Exploring Spatio-Temporal Graph Convolution for Video-based Human-Object Interaction Recognition"

### Installation
1. Clone this repository.   
    ```
    git clone https://github.com/NingWang2049/STIGPN2.git
    ```
  
2. Install Python dependencies:   
    ```
    pip install -r requirements.txt
    ```
 ### Prepare Data
 1. Follow [something-else repository](https://github.com/joaanna/something_else) to prepare the data of something-else dataset.
 2. Process the data using data/fatch.py
 3. We provide some checkpoints to the trained models. Download them [here](https://drive.google.com/drive/folders/1oD1fdLx09kLJc4XF9-VryO6SZC9_VkAj?usp=sharing) and put them in the checkpoints folder
### Testing
For the Something-else dataset:
    ```
        python eval.py
    ```
### Training
    ```
        python train.py --model V_SSTGCN
    ```
    ```
        python train.py --model SSTGCN 
    ```
