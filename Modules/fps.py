import pandas as pd
import sys

# assumes that core oracle filters down to specific batter
def main():
    no_swing = ['Ball', 'Called Strike']
    take = 0
    swing = 0
    df = pd.read_csv(sys.stdin)
    at_bat_id = -1
    for index, row in df.iterrows():
        # first pitch
        if at_bat_id != row["at_bat_id"]:
            if row["des"] in no_swing:
                take+=1
            else:
                swing+=1
            at_bat_id = row["at_bat_id"]

            

    fps = float(swing)/float(take + swing)
    sys.stdout.write("%.3f" % fps)


if __name__ == "__main__":
    main()
