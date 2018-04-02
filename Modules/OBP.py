import pandas as pd
import sys

def main():
    onBase = ["Single", "Double", "Triple", "Home Run", "Walk", "Hit By Pitch", "Intent Walk"]
    atBat = ["Single", "Double", "Triple", "Home Run", "Field Out",
    "Strikeout", "Grounded Into Double Play", "Force Out", "Hit By Pitch", "Sac Fly"
    "Walk", "Intent Walk", "Sac Bunt"]
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
