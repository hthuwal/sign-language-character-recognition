sudo apt-get install python-pip python-dev
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL

pip install Cython
pip install pandas


sudo apt-get install python-numpy python-scipy ##pip install numpy ##pip install scipy

pip install -U scikit-learn

sudo apt-get install python-skimage

pip install tqdm

pip install pyyaml
sudo apt-get install libhdf5-serial-dev

pip install keras

sudo apt-get install unzip
pip install h5py

sudo apt-get install libopencv-dev python-opencv ##pip install opencv-python

echo "Change image_dim_ordering: 'tf' to image_dim_ordering: 'th' in your ~/.keras/keras.json"