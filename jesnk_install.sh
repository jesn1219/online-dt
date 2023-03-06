# https://github.com/jesn1219/online-dt

sudo apt-get install libghc-x11-dev
sudo apt-get install libglew-dev
sudo apt-get install patchelf

pip install -U transfomrers==4.6
pip install icecream
python ./data/download_d4rl_antmaze_datasets.py
python ./data/download_d4rl_gym_datasets.py

# Add follow code to ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jskang/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jskang/anaconda3/envs/odt/lib


# tensorboard
tensorboard --logdir exp --bind_all --port 6006
