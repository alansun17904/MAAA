# in development - not sure if this works
BRANCH_NAME="test"
ENV_NAME="MAAA_CD"
CLONE=true

apt-get update
apt-get upgrade

apt-get install unzip

# Install Mini conda
# make this optional / argumented
# make this an arg
mkdir -p ./miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
rm ./miniconda3/miniconda.sh
source ./miniconda3/bin/activate
conda init --all
source ~/.bashrc

git clone https://github.com/alansun17904/MAAA.git
cd MAAA
git checkout $BRANCH_NAME

conda env create -f environment.yml
conda activate $ENV_NAME

unzip data.zip