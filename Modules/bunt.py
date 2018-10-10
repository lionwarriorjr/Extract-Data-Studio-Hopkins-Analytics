import pandas as pd
import sys

# assumes that core oracle filters down to specific batter
def main():
    bunt = 0
    at_bat = 0
    df = pd.read_csv("2017.csv")
    at_bat_id = -1
    for index, row in df.iterrows():
        batter_id = row["batter"]
        if at_bat_id != row["at_bat_id"]:
            if "bunt" in row["atbat_des"]:
                bunt+=1
            at_bat+=1
            at_bat_id = row["at_bat_id"]

            

    steal = float(bunt)/float(at_bat)
    sys.stdout.write("%.3f" % steal)


if __name__ == "__main__":
    main()
