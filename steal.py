import pandas as pd
import sys
import collections

def main():
    swings = collections.Counter()
    #df = pd.read_csv(sys.stdin)
    df = pd.read_csv("2017.csv")
    batter_up = ""
    total = 0
    successful_steal = 0
    attempted_steal = 0
    batter = "DJ Peters"

    for index, row in df.iterrows():
        # new bat
        if batter_up != row["at_bat_id"]:
            # sucessfully stole a base
            if batter + " steals" in row["atbat_des"]:
                successful_steal+=1
            # got caught stealing
            if batter + " caught stealing" in row["atbat_des"]:
                attempted_steal+=1
            total+=1
        batter_up = row["at_bat_id"]
           
    # percentage of stealing or attempting to steal
    perc_steal = (successful_steal + attempted_steal)/total
    sys.stdout.write("%.3f" % perc_steal)


if __name__ == "__main__":
    main()