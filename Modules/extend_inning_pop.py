import pandas as pd
import sys

def main():
    bat_type = "popup"
    out_type = "pops out"
    extend_inning = 0
    total = 0
    ball = False
    batter_up = 0
    inning = 0
    df = pd.read_csv(sys.stdin)
    #df = pd.read_csv("2017.csv")
    for index, row in df.iterrows():
        if ball == True:
            if batter_up != row["at_bat_id"]: # make sure it was last pitch of the at bat
                if inning == row["inning_side"]: # make sure the inning is still the same
                    extend_inning+=1
                total+=1
            ball = False
        if row["outs_when_up"] == 2:
            if (row["bb_type"] == bat_type or row["bb_type"] == out_type) and (row["on_2b"] != "NA" or row["on_3b"] != "NA"): # runner in scoring position
                ball = True
                batter_up = row["at_bat_id"]
                inning = row["inning_side"]

    extend = float(extend_inning)/float(total)
    sys.stdout.write("%.3f" % extend)


if __name__ == "__main__":
    main()
