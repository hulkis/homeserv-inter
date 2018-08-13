echo "Runnin env"
env  # print out env

echo "Running ls"
ls

# Fix pbr (.git not clonned during floyd job)
git init . && git add . && git commit -m "floyd commit"

pip install -U pip
pip install -r requirements.txt
pip install --editable .

# sym link to dataset:
pkg_dir=$PWD
ln -s /data .

# install ml_utils
git clone https://github.com/gjeusel/ml_utils
(cd ml_utils && pip install -e .)

# install wax-toolbox
git clone https://github.com/gjeusel/wax-toolbox
(cd wax-toolbox && pip install -e .)
