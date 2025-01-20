import re

if __name__ == "__main__":
    RECORDING_RC = False
    RECORDING_MDS = False
    RC = []
    MDS = []
    with open("/home/thomas/secure-data-valuation/ezkl/src/circuit/modules/poseidon/poseidon_params.rs", "r") as f:
        lines = f.readlines()
        for line in lines:
            if "ROUND_CONSTANTS" in line and "pub(crate)" in line:
                RECORDING_RC = True
                continue
            if "//" in line and RECORDING_RC:
                RECORDING_RC = False
            if "MDS" in line and "pub(crate)" in line:
                RECORDING_MDS = True
                continue
            if "//" in line and RECORDING_MDS:
                RECORDING_MDS = False
                continue
            if RECORDING_RC and "0x" in line:
                line = line.strip()
                line = line.replace("_", "")
                line = line.replace(",", "")
                RC.append(line)
            if RECORDING_MDS and "0x" in line:
                line = line.strip()
                line = line.replace("_", "")
                line = line.replace(",", "")
                MDS.append(line)
    RC_update = []
    for a,b in zip(RC[0::2], RC[1::2]):
        RC_update.append(f"0x{a[2:]}{b[2:]}")
    RC = RC_update
    with open("poseidon_params.py", "w") as f:
        f.write("RC = [")
        for i in range(len(RC)):
            f.write("'"+RC[i]+"'")
            if i != len(RC) - 1:
                f.write(", ")
        f.write("]\n")
        f.write("MDS = [")
        for i in range(4):
            f.write("[")
            for j in range(4):
                f.write("'"+MDS[i * 4 + j]+"'")
                if j != 3:
                    f.write(", ")
            f.write("]")
            if i != 3:
                f.write(", ")
        f.write("]\n")

