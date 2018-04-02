import pandas as pd
import sys

def main():
    hit = ["Single", "Double", "Triple"]
    atBat = ["Single", "Double", "Triple", "Field Out",
    "Grounded Into Double Play", "Force Out", "Sac Fly", "Sac Bunt"]
    hits = 0
    ab = 0
    df = pd.read_csv(sys.stdin)
    # get relevant data
    for index, row in df.iterrows():
        if row["event"] in atBat:
            ab += 1
            if row["event"] in hit:
                hits += 1
    babip = hits / ab
    sys.stdout.write("%.3f" % babip)



if __name__ == "__main__":
    main()
