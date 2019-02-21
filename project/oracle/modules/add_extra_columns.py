import csv
import os

with open('pitchfx2017.csv','r') as csvinput:
    with open('2017_updated1.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('outs_when_up')
        all.append(row)
        outs = 0
        second_outs = 0
        inning = "top"
        batter = 0
        for row in reader:
            if outs != row[5]:
                if row[18] != inning:
                    outs = 0
                    inning = row[18]
                second_outs = outs
            elif batter != row[1]:
                second_outs = outs
            row.append(str(second_outs))
            all.append(row)
            outs = row[5]
            batter = row[1]

        writer.writerows(all)


with open('2017_updated1.csv','r') as csvinput:
    with open('2017_updated2.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('home_team_runs_when_up')
        row.append('away_team_runs_when_up')
        all.append(row)
        home = 0
        away = 0
        second_home = 0
        second_away = 0
        batter = 0
        for row in reader:
            if batter != row[1]:
                if second_home != row[15]:
                    second_home = home
                if second_away != row[16]:
                    second_away = away
            row.append(str(second_home))
            row.append(str(second_away))
            all.append(row)
            home = row[15]
            away = row[16]
            batter = row[1]

        writer.writerows(all)

with open('2017_updated2.csv','r') as csvinput:
    with open('2017_updated3.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('bb_type')
        all.append(row)
        for row in reader:
            if "sac fly" in row[13].lower():
                row.append("sac_fly")
            elif "sac bunt" in row[13].lower():
                row.append("sac_bunt")
            elif "fly ball" in row[11].lower():
                row.append("fly_ball")
            elif "ground ball" in row[11].lower():
                row.append("ground_ball")
            elif "line drive" in row[11].lower():
                row.append("line_drive")
            elif "popup" in row[11].lower():
                row.append("popup")
            elif "flies out" in row[11].lower():
                row.append("fly_out")
            elif "grounds out" in row[11].lower():
                row.append("grounds_out")
            elif "lines out" in row[11].lower():
                row.append("lines_out")
            elif "pops out" in row[11].lower():
                row.append("pop_out")
            elif "walks" in row[11].lower():
                row.append("walk")
            else:
                row.append("strike_out")
            all.append(row)

        writer.writerows(all)

with open('2017_updated3.csv','r') as csvinput:
    with open('2017_updated4.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('at_bat_id')
        all.append(row)
        batter = 0
        id_ = 0
        for row in reader:
            if batter != row[4]:
                id_+=1
                batter = row[4]
            row.append(str(id_))
            all.append(row)

        writer.writerows(all)


with open('2017_updated4.csv','r') as csvinput:
    with open('2017_updated.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('in_strike_zone')
        all.append(row)
        strike = 0
        for row in reader:
            if row[45] == "NA":
                strike = 0
            elif (float(row[45]) > 0.8391667) or (float(row[45]) < -0.8391667):
                strike = 0
            elif (float(row[46]) < float(row[54])) or (float(row[46]) > float(row[53])):
                strike = 0
            else:
                strike = 1
            row.append(str(strike))
            all.append(row)

        writer.writerows(all)

os.remove("2017_updated1.csv")
os.remove("2017_updated2.csv")
os.remove("2017_updated3.csv")
os.remove("2017_updated4.csv")