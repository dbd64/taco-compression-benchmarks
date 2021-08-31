#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclude=/data/scratch/danielbd/node_exclusions.txt
#SBATCH --exclusive

set -u


root=/data/scratch/danielbd/clips/
declare -a folders=("ed_1600/scene1/" "ed_1600/scene2/" "ed_1600/scene3/" "ed_1600/scene4/" "sita/intro_flat/" "sita/images_background/" "sita/flat_textures/" "sita/drawn_flat/" "stock_videos/paperclips/" "stock_videos/pink_bars/" "stock_videos/ppt/" "stock_videos/rect_bkgd/")
out_file=/data/scratch/danielbd/clips/imgs.log

# root=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/data
# declare -a folders=("ed/scene1")
# out_file=/Users/danieldonenfeld/Developer/taco-compression-benchmarks/img_list.log

:> $out_file

print_file_name(){
    if [ -f "$1" ]; then
        echo "$1" >> $out_file
    else 
        echo "$1 does not exist."
    fi
}

for folder in "${folders[@]}"
do
    folder="$root/$folder"

    for file in "$folder/00"{1..9}".png"; do
        print_file_name $file
    done

    for file in "$folder/0"{10..99}".png"; do
        print_file_name $file
    done

    for file in "$folder/"{100..340}".png"; do
        print_file_name $file
    done
done

echo >> $out_file