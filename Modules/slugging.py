import pandas as pd
import sys

def main():
    atBat = ["single", "double", "triple", "home_run", "field_out",
    "strikeout", "grounded_into_double_play", "force_out"]
    hits = 0
    ab = 0
    df = pd.read_csv(sys.stdin)
    # get relevant data
    for index, row in df.iterrows():
        if row["event"] in atBat:
            ab += 1
            if row["event"] == "single":
                hits += 1
            if row["event"] == "double":
                hits += 2
            if row["event"] == "triple":
                hits += 3
            if row["event"] == "home_run":
                hits += 4
    slg = hits / ab
    sys.stdout.write("%.3f" % slg)



if __name__ == "__main__":
    main()
