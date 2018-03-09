import pandas as pd
import sys

def main():
    onBase = ["single", "double", "triple", "home_run", "walk", "hit_by_pitch", "intent_walk"]
    atBat = ["single", "double", "triple", "home_run", "field_out",
    "strikeout", "grounded_into_double_play", "force_out", "hit_by_pitch", "sac_fly"
    "walk", "intent_walk"]
    ob = 0
    ab = 0
    df = pd.read_csv(sys.stdin)
    # get relevant data
    for index, row in df.iterrows():
        if row["event"] in atBat:
            ab += 1
            if row["event"] in onBase:
                ob += 1

    obp = ob / ab
    sys.stdout.write("%.3f" % obp)



if __name__ == "__main__":
    main()
