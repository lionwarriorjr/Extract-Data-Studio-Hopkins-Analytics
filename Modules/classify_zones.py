import pandas as pd
import sys
import collections
import numpy as np

def main():
    pitches = {}
    # make each zone have a dictionary (this will be pitch_type to how many misses/hits they get)
    for i in range(1, 38):
        pitches[i] = {}

    #df = pd.read_csv(sys.stdin)
    df = pd.read_csv("2017.csv")
    ball_type = ""
    zone = ""


    for index, row in df.iterrows():
        # classify zone
        x = row["px"]
        z = row["pz"]
        if x<-1 or x>1:
            horiz = 9
        if -1<x<-0.75:
            horiz = 1
        if -0.75<x<-0.5:
            horiz = 2
        if -0.5<x<-0.25:
            horiz = 3
        if -0.25<x<0:
            horiz = 4
        if 0<x<0.25:
            horiz = 5
        if 0.25<x<0.5:
            horiz = 6
        if 0.5<x<0.75:
            horiz = 7
        if 0.75<x< 1.0:
            horiz = 8

        if z<1.5 or z>3.0:
            vert = 9
        if 1.5<z<1.75:
            vert = 1
        if 1.75<z<2.0:
            vert = 2
        if 2.0<z<2.25:
            vert = 3
        if 2.25<z<2.5:
            vert = 4
        if 2.5<z<2.75:
            vert = 5
        if 2.75<z<3.0:
            vert = 6
        if 3.0<z<3.25:
            vert = 7
        if 3.25<z<3.5:
            vert = 8

        
        if (horiz,vert) == (2,7):
            zone = 1
        if (horiz,vert) == (3,7):
            zone = 1
        if (horiz,vert) == (2,6):
            zone - 1
        if (horiz,vert) == (3,6):
            zone = 1


        if (horiz,vert) == (4,7):
            zone = 2
        if (horiz,vert) == (5,7):
            zone = 2
        if (horiz,vert) == (4,6):
            zone = 2
        if (horiz,vert) == (5,6):
            zone = 2


        
        if (horiz,vert) == (6,7):
            zone = 3
        if (horiz,vert) == (7,7):
            zone = 3
        if (horiz,vert) == (6,6):
            zone = 3
        if (horiz,vert) == (7,6):
            zone = 3

        
        if (horiz,vert) == (6,5):
            zone = 4
        if (horiz,vert) == (7,5):
            zone = 4
        if (horiz,vert) == (6,4):
            zone = 4
        if (horiz,vert) == (7,4):
            zone = 4


        
        if (horiz,vert) == (4,5):
            zone = 5
        if (horiz,vert) == (5,5):
            zone = 5
        if (horiz,vert) == (4,4):
            zone = 5
        if (horiz,vert) == (5,4):
            zone = 5


        
        if (horiz,vert) == (2,5):
            zone = 6
        if (horiz,vert) == (3,5):
            zone = 6
        if (horiz,vert) == (2,4):
            zone = 6
        if (horiz,vert) == (3,4):
            zone = 6


        
        if (horiz,vert) == (2,2):
            zone = 7
        if (horiz,vert) == (2,3):
            zone = 7
        if (horiz,vert) == (3,2):
            zone = 7
        if (horiz,vert) == (3,3):
            zone = 7


        
        if (horiz,vert) == (4,3):
            zone = 8
        if (horiz,vert) == (5,3):
            zone = 8
        if (horiz,vert) == (4,2):
            zone = 8
        if (horiz,vert) == (5,2):
            zone = 8


        
        if (horiz,vert) == (6,3):
            zone = 9
        if (horiz,vert) == (7,3):
            zone = 9
        if (horiz,vert) == (6,2):
            zone = 9
        if (horiz,vert) == (7,2):
            zone = 9


        
        if (horiz,vert) == (1,8):
            zone = 10
        if (horiz,vert) == (2,8):
            zone = 11
        if (horiz,vert) == (3,8):
            zone = 12
        if (horiz,vert) == (4,8):
            zone = 13
        if (horiz,vert) == (5,8):
            zone = 14
        if (horiz,vert) == (6,8):
            zone = 15
        if (horiz,vert) == (7,8):
            zone = 16
        if (horiz,vert) == (8,8):
            zone = 17
        if (horiz,vert) == (8,7):
            zone = 18
        if (horiz,vert) == (8,6):
            zone = 19
        if (horiz,vert) == (8,5):
            zone = 20
        if (horiz,vert) == (8,4):
            zone = 21
        if (horiz,vert) == (8,3):
            zone = 22
        if (horiz,vert) == (8,2):
            zone = 23
        if (horiz,vert) == (8,1):
            zone = 24
        if (horiz,vert) == (7,1):
            zone = 25
        if (horiz,vert) == (6,1):
            zone = 26
        if (horiz,vert) == (5,1):
            zone = 27
        if (horiz,vert) == (4,1):
            zone = 28
        if (horiz,vert) == (3,1):
            zone = 29
        if (horiz,vert) == (2,1):
            zone = 30
        if (horiz,vert) == (1,1):
            zone = 31
        if (horiz,vert) == (1,2):
            zone = 32
        if (horiz,vert) == (1,3):
            zone = 33
        if (horiz,vert) == (1,4):
            zone = 34
        if (horiz,vert) == (1,5):
            zone = 35
        if (horiz,vert) == (1,6):
            zone = 36
        if (horiz,vert) == (1,7):
            zone = 37

        if horiz == 9 or vert == 9:
            zone = 50

        # classify if this is a swing_miss, hit_out, or hit. Otherwise we don't care about the pitch
        if row["type"] == "S" and (row["des"] == "Foul Strike" or row["des"] == "Foul Tip Strike" or row["des"] == "Swinging Strike"):
            ball_type = "swing_miss"
        elif row["type"] == "X":
            if "out" in row["event"]:
                ball_type = "out"
            else:
                ball_type = "hit"
        else:
            ball_type = "none"

        # Was having issues reading in null values, so this is a work around
        if isinstance(row["pitch_type"], str):
            pitch_type = row["pitch_type"]
        else:
            ball_type = "none"

        # again this just makes sure we're only adding pitches we care about
        if ball_type != "none" and zone != 50:
            # this just adds the pitch_type dictionary if it doesn't already exist
            if pitch_type not in pitches[zone]:
                pitches[zone][pitch_type] = collections.Counter()
            # Since we created a counter, we can incremement it without declaring a new dictionary
            pitches[zone][pitch_type][ball_type]+=1


    # goes through each zone and pitch_type in that zone and sees if it should be green, yellow, red or nothing.
    output = {}
    for zone in pitches:
        output[zone] = []
        for pitch_type in pitches[zone]:
            hits = pitches[zone][pitch_type]['hit']
            outs = pitches[zone][pitch_type]['out']
            swing_miss = pitches[zone][pitch_type]['swing_miss']
            total = hits + outs + swing_miss
            if total != 0:
                if swing_miss/total > .8:
                    output[zone].append("green " + pitch_type)
                elif (swing_miss + outs)/total > .8:
                    output[zone].append("yellow " + pitch_type)
                elif hits/total > .8:
                    output[zone].append("red " + pitch_type)

    sys.stdout.write(str(output))


if __name__ == "__main__":
    main()