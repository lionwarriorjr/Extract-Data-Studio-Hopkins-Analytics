import pandas as pd
import sys

def main():
    hit = ["Single", "Double", "Triple", "Home Run"]
    atBat = ["Single", "Double", "Triple", "Home Run", "Field Out",
    "Strikeout", "Grounded Into Double Play", "Force Out"]
    hits = 0
    ab = 0
    df = pd.read_csv(sys.stdin)
    # get relevant data
    for index, row in df.iterrows():
        if row["event"] in atBat:
            ab += 1
            if row["event"] in hit:
                hits += 1
    avg = hits / ab
    sys.stdout.write("%.3f" % avg)



if __name__ == "__main__":
    main()
