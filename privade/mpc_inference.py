import os

#Current directory
cur_dir = os.path.dirname(os.path.abspath(__file__))


def setup(name):
    os.system(f"cd {cur_dir}/../MP-SPDZ && ./compile.py -R 64 {name}")
    
    
    
def inference(name,fake_offline=False):
    if fake_offline:
        os.system(f"cd {cur_dir}/../MP-SPDZ && ./Fake-Offline.x 2 -Z 64 -S 48 -p {name}")

    os.system(f"cd {cur_dir}/../MP-SPDZ && Scripts/spdz2k.sh {name}")