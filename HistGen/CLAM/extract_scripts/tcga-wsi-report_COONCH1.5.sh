#!/bin/bash
DIR_TO_COORDS="/home/woody/iwi5/iwi5204h/HistGen/Data/WSI/PatchResults_UNI2/"
DATA_DIRECTORY="/home/janus/iwi5-datasets/REG2025/Train_01/"
CSV_FILE_NAME="/home/woody/iwi5/iwi5204h/HistGen/Data/Label/output.csv"
FEATURES_DIRECTORY=$DIR_TO_COORDS
ext=".tiff"
save_storage="yes"
root_dir="/home/woody/iwi5/iwi5204h/HistGen/CLAM/extract_scripts/logs/UNI2/"

# models="resnet50"
# models="ctranspath"
# models="plip"
#models="dinov2_vitl"
#models="uni1"
#models="uni2"
models="coonch1.5"

declare -A gpus
gpus["resnet50"]=0
gpus["resnet101"]=0
gpus["ctranspath"]=0
gpus["dinov2_vitl"]=0
gpus['plil']=0
gpus['uni1']=0
gpus['uni2']=0
gpus['coonch1.5']=0

#datatype="tcga" # extra path process for TCGA dataset, direct mode do not care use extra path
datatype="reg2025" # extra path process for TCGA dataset, direct mode do not care use extra path

for model in $models
do
        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        python3 /home/woody/iwi5/iwi5204h/HistGen/CLAM/extract_features_fp.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 8 \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage > $root_dir$model".txt" 2>&1

done