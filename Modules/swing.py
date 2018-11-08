import pandas as pd
import sys
import collections

def main():
    swings = collections.Counter()
    #df = pd.read_csv(sys.stdin)
    df = pd.read_csv("2017.csv")
    batter_up = ""
    swing = ""
    total = 0
    event = ""

    for index, row in df.iterrows():
        # new batter, we want to look at what happened at last pitch
        if batter_up != row["at_bat_id"]:
            if event in ["Single", "Double", "Triple", "Groundout", "Flyout", "Lineout", "Forceout", "Home Run", "Grounded into DP", "Popout"]:
                # incrememnts that swing - could have multiple swings as declared below
                # s is each of the swing that we declared (if there is just one)
                # this foor loop will only run once.
                for s in swing:
                    swings[s]+=1
                    # increment total swings
                    total+=1
            batter_up = row["at_bat_id"]
        # parse where to swing
        swing = ""
        for word in row["atbat_des"].split(" "):
            if word.lower() in ["left", "third"]:
                # set a specific value for whatever kind of swing it was
                swing = ["Left hitter"]
                # now that we have found the word, we can exit out of this
                # loop and stop looking for more words
                break
            elif word.lower() in ["pitcher", "center"]:
                swing = ["Middle hitter"]
                break
            elif word.lower() in ["first", "right"]:
                swing = ["Right hitter"]
                break
            elif word.lower() == "shortstop":
                swing = ["Left hitter", "Middle hitter"]
            elif word.lower() == "second":
                swing = ["Middle hitter", "Right hitter"]
        event = row["event"]

    hitter = "spray hitter"

    most_common_swing = swings.most_common(1)[0]
    # check out thresholds
    if float(most_common_swing[1])/total > .6:
        hitter = most_common_swing[0]

    # most likely to hit this way
    sys.stdout.write(hitter)


if __name__ == "__main__":
    main()
