import pandas as pd
import sys
import collections

def main():
    pitches = collections.Counter()
    #df = pd.read_csv(sys.stdin)
    df = pd.read_csv("2017.csv")
    batter_up = -1
    event = ""
    second_last_at_bat_pitch_type = ""
    last_at_bat_pitch_type = ""

    for index, row in df.iterrows():
        # new batter, we want to look at what happened at last pitch
        if batter_up != row["at_bat_id"]:
            # we only care if this guy struck out
            if event == "Strikeout":
                power_sequence = str(second_last_at_bat_pitch_type) + " " + str(last_at_bat_pitch_type)
                # if we've seen this sequence before, we incrememnt by 1.
                # if we haven't, default is 0
                pitches[power_sequence]+=1
            second_last_at_bat_pitch_type = ""
            last_at_bat_pitch_type = ""
            event = ""
            batter_up = row["at_bat_id"]
        else:
            # we want last two at bats at all times in case they are the last pitch
            # puts them into zone based on four squares - zone 1-4.
            # possbile TODO: account for right/left hitters
            x = row["px"]
            z = row["pz"]
            if x < 1.25 or x > 1.25:
                horiz = 3
            if -1.25 < x < 0:
                horiz = 1
            if 0 < x < 1.25:
                horiz = 2
            if z<0.75 or z>3.58:
                vert = 3
            if 0.75 < z < 2.176:
                vert = 1
            if 2.17 < z < 3.58:
                vert = 2
            if (horiz,vert) == (1,1):
                zone = 'Zone 3'
            if (horiz,vert) == (2,1):
                zone = 'Zone 4'
            if (horiz,vert) == (1,2):
                zone = 'Zone 2'
            if (horiz,vert) == (2,2):
                zone = 'Zone 1'
            if horiz == 3 or vert == 3:
                zone = "ball"
            second_last_at_bat_pitch_type = last_at_bat_pitch_type
            last_at_bat_pitch_type = str(row["pitch_type"]) + " " + str(zone)
            event = row["event"]

    # best power sequence
    best_power_sequence = max(pitches.items(), key=lambda k: k[1])
    sys.stdout.write(str(pitches.most_common(1)[0][0]))


if __name__ == "__main__":
    main()