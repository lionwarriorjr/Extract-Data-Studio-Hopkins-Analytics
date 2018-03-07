import pandas as pd
import sys

def main():
    hit = ["single", "double", "triple"]
    atBat = ["single", "double", "triple", "field_out",
    "grounded_into_double_play", "force_out", "sac_fly"]
    hits = 0
    ab = 0
    df = pd.read_csv(sys.stdin)
    # get relevant data
    for index, row in df.iterrows():
        if row["events"] in atBat:
            ab += 1
            if row["events"] in hit:
                hits += 1
    babip = hits / ab
    sys.stdout.write("%.3f" % babip)



if __name__ == "__main__":
    main()
