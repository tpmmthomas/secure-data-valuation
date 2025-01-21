import re

def reverse_hex_string(hex_string):
    hexstr = hex_string[2:]
    byte_list = [hexstr[i:i+2] for i in range(0, len(hexstr), 2)]
    reversed_hex = "".join(reversed(byte_list))
    return "0x" + reversed_hex

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
                continue
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
                #Reverse the order of bytes
                # line = reverse_hex_string(line)
                RC.append(line)
            if RECORDING_MDS and "0x" in line:
                line = line.strip()
                line = line.replace("_", "")
                line = line.replace(",", "")
                # line = reverse_hex_string(line)
                MDS.append(line)
    RC_update = []
    for a,b,c,d in zip(RC[0::4], RC[1::4], RC[2::4], RC[3::4]):
        RC_update.append(f"0x{a[2:]}{b[2:]}{c[2:]}{d[2:]}")
        # RC_update.append(f"0x{d[2:]}{c[2:]}{b[2:]}{a[2:]}")
    RC = RC_update
    MDS_update = []
    for a,b,c,d in zip(MDS[0::4], MDS[1::4], MDS[2::4], MDS[3::4]):
        MDS_update.append(f"0x{a[2:]}{b[2:]}{c[2:]}{d[2:]}")
        # MDS_update.append(f"0x{d[2:]}{c[2:]}{b[2:]}{a[2:]}")
    MDS = MDS_update
    with open("poseidon_params.py", "w") as f:
        f.write("RC = [")
        for i in range(len(RC)):
            f.write("'"+RC[i]+"'")
            if i != len(RC) - 1:
                f.write(", ")
        f.write("]\n")
        f.write("MDS = [")
        for i in range(2):
            f.write("[")
            for j in range(2):
                f.write("'"+MDS[i * 2 + j]+"'")
                if j != 1:
                    f.write(", ")
            f.write("]")
            if i != 3:
                f.write(", ")
        f.write("]\n")

