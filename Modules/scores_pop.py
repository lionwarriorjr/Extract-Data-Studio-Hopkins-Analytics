import pandas as pd
import sys

def main():
    bat_type = "popup"
    out_type = "pop_out"
    total = 0 # total number of bb_type hit
    ball = False
    batter_up = 0
    score = 0
    runs = 0
    df = pd.read_csv(sys.stdin)
    #df = pd.read_csv("2017.csv")
    for index, row in df.iterrows():
        if ball == True:
            if batter_up != row["at_bat_id"]: #confirms we were on last pitch
                if score == "T": # if true, this scored a run
                    runs+=1 # we scored a run from this bb_type
                    score = 0
                total+=1
            ball = False
        if row["outs_when_up"] == 2:
            if (row["bb_type"] == bat_type or row["bb_type"] == out_type) and (row["on_2b"] != "NA" or row["on_3b"] != "NA") : # runner in scoring position
                ball = True # this bat needs to be examined
                batter_up = row["at_bat_id"] # track to make sure we are on last bat
                score = row["score"] # see if we scored a run or not
    extend = float(runs)/float(total)
    sys.stdout.write("%.3f" % extend)


if __name__ == "__main__":
    main()
