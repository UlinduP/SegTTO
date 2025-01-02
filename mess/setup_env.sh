# run script with
# bash mess/setup_env.sh

# Create new environment "catseg"
conda create --name catseg -y python=3.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate catseg

# install requirements from CAT-Seg
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt

cd open_clip/ && make install

# install packages for dataset preparation
pip install gdown
pip install kaggle
pip install rasterio
pip install pandas