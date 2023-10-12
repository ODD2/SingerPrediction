# Artist20 Singer Prediction
## Description
This repository includes the project for the first homework of the course "Deep Learning for Music Analysis and Generation" lectured by Prof. Yang at the National Taiwan University. The main goals of this work is to train a singer classification model on the [Artist20](http://labrosa.ee.columbia.edu/projects/artistid/) dataset. Given an audio segment, the model should predict the top 3 highest similarity artist within the 20 different singers in the dataset.
## Create Environment 
```bash
conda create env -f environment.yml
conda activate sing_id
```
## Preprocess Data with Demucs
Please follow the instructions in the '**Calling From Another Python Program**' section to process all the audios.
https://github.com/facebookresearch/demucs#calling-from-another-python-program

The file structure should be something similar like this:
```
./dataset
    |- test/
        |- 0001/
            |- vocals.mp3
        |- 0002/
            |- vocals.mp3
        |- 0003/
            |- vocals.mp3
        ...
    | - train/
        |- aerosmith/
            |- Aerosmith/
                |- 01-Make_it/
                    |- vocals.mp3
        |- $singer_n/
            |- $album_n/
                |- $song_n/
                    |- $audios_n
        ...
    | - valid/
        |- $singer_n/
            |- $album_n/
                |- $song_n/
                    |- $audios_n
        ...
        
```
# Inference the Audios
 - Please download the model weights from Google Drive: [Link](https://drive.google.com/file/d/146NvIeEPlIJ9VlO9GYA5ZpAVwTa3gcj9/view?usp=drive_link)
 - Please download the singer anchors from Google Drive: [Link](https://drive.google.com/file/d/1eMBlQElQLIM7esmzLMa5UiM21F7z4WZh/view?usp=drive_link)
 - Inference the model with the following command: 
 ```shell
 python -m inference \
    --anchor_path="./singer_samples.pickle" \ # the path to singer anchors
    --weight_path="./model_weight.pt" \ # the path to the weights
    --out_path="./output.csv" \ # the path to dump the inference results
    --root_dir="./dataset/test/" \ # the root folder path
    --glob_exp="*/vocals.mp3" \ # the glob expression to search the root folder
    --duration=20 \ # the duration of audio segment to inference 
    --batch_size=32 # the maximum batch size
 ```


    

