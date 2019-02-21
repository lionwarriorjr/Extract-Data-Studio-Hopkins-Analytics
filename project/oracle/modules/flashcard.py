import numpy as np
from scipy import stats
import pandas as pd
import json
import collections
from oracle.modules.steal import Steal
from oracle.modules.bunt import Bunt
from oracle.modules.swing import Swing
from oracle.modules.fpt import FPT
from oracle.modules.fps import FPS
from oracle.modules.power_sequence import PowerSequence
from oracle.modules.zones import ClassifyZones
from oracle.modules.module import Module
from PIL import Image, ImageDraw, ImageFont

class Flashcard(Module):

    def set_module(self):
        return False
    
    def get_lexicon(self):
        result = set()
        result.add('flashcard')
        return result
    
    def run_flashcard(self, batting, appearances, batter):
        print('evaluating flashcard module')
        flashcard = {}
        flashcard['batter'] = batter
        stand = str(batting.loc[batting.index[0],'stand'])
        flashcard['stand'] = stand
        steal = Steal()
        perc_steal = steal.execute(range(len(appearances.index)), appearances, {'batter': batter}, False) 
        flashcard['perc_steal'] = perc_steal.iloc[0,0]
        bunt = Bunt()
        bunt_pct = bunt.execute(range(len(batting.index)), batting, {}, False)
        flashcard['bunt'] = bunt_pct.iloc[0,0]
        fps = FPS()
        fps_ratio = fps.execute(range(len(batting.index)), batting, {}, False)
        flashcard['fps'] = fps_ratio.iloc[0,0]
        fpt = FPT()
        fpt_ratio = fpt.execute(range(len(batting.index)), batting, {}, False)
        flashcard['fpt'] = fpt_ratio.iloc[0,0]
        swing = Swing()
        swing_pct = swing.execute(range(len(batting.index)), batting, {}, False)
        flashcard['swing_type'] = swing_pct.iloc[0,0]
        power = PowerSequence()
        power_sequence = power.execute(range(len(batting.index)), batting, {}, False)
        flashcard['powerSequence'] = power_sequence.iloc[0,0].strip()
        classify_zones = ClassifyZones()
        zones = classify_zones.execute(range(len(batting.index)), batting, {}, False)
        flashcard['zones'] = zones
        with open('flashcard.json', 'w') as outfile:
            json.dump(flashcard, outfile)
        # call visualization subroutine
        filename = self.generate_flashcard_visualization()
        return filename
    
    '''if self.is_filter return an index set else return a table'''
    def execute(self, iset, data, params={}, is_grouped=False):
        if is_grouped:
            return None
        df = data.iloc[iset,:]
        if data.shape[0] > 0:
            batter = params['batter']
            appearances = df
            df = df[df['batter_name'] == batter]
            if df.shape[0] > 0:
                batting_left = df[df['stand'] == 'L']
                batting_right = df[df['stand'] == 'R']
                if batting_left.shape[0] > 0 and batting_left.shape[0] > 0.2 * df.shape[0]:
                    return self.run_flashcard(batting_left, appearances, batter)
                if batting_right.shape[0] > 0 and batting_right.shape[0] > 0.2 * df.shape[0]:
                    return self.run_flashcard(batting_right, appearances, batter)
    
    '''helper code to generate Flashcard visualization'''
    def generate_flashcard_visualization(self):
        with open('flashcard.json') as data_file:
            data = json.load(data_file)

        side = data['stand']

        side_width = 600
        side_height = 800
        font = ImageFont.truetype("Butler_Regular.ttf", 15)

        img = Image.new('RGB', (side_width, side_height), color = 'white')
        d = ImageDraw.Draw(img)

        if side == "R":
            player = Image.open('right.jpg')
            player.thumbnail((150,150))
            img.paste(player, (450,250))
        else:
            player = Image.open('left.jpg')
            player.thumbnail((150,150))
            img.paste(player, (20,250))

        board_length = 200
        small_box_length = 25
        median_box_length = 50
        dist_to_edge = (side_width - board_length)/2
        dist_to_top = (side_height - board_length)/3

        for i in range (0,board_length,small_box_length):
            d.rectangle([dist_to_edge +i ,dist_to_top,dist_to_edge +i + small_box_length,dist_to_top + small_box_length], outline="black")

        for j in range (small_box_length, 7*small_box_length, small_box_length):
            d.rectangle([dist_to_edge, dist_to_top + j,dist_to_edge + small_box_length, dist_to_top + j + small_box_length], outline="black")

        for k in range (0,board_length,small_box_length):
            d.rectangle([dist_to_edge +k ,dist_to_top + 7 * small_box_length ,dist_to_edge + k + small_box_length, dist_to_top + 8 * small_box_length], outline="black")

        for l in range (small_box_length, 7*small_box_length, small_box_length):
            d.rectangle([dist_to_edge + 7*small_box_length, dist_to_top + l,dist_to_edge + 8*small_box_length, dist_to_top + l + small_box_length], outline= "black")

        for m in range(0,3):
            for n in range(0,3):
                d.rectangle([dist_to_edge+small_box_length + m*median_box_length, dist_to_top + small_box_length + n*median_box_length, dist_to_edge + small_box_length + m*median_box_length + median_box_length, dist_to_top + small_box_length + n*median_box_length + median_box_length], outline="black")


        list = ImageDraw.Draw(img)

        top = 10
        for x in data:
            if x == "zones" or x == "stand":
                continue
            list.text((20, top), x + " : " + data[x], fill=(0, 0, 0), font=font)
            top = top + 20

        circles_string = data['zones']
        convert_string = json.loads(circles_string)

        input = convert_string.get('zones')

        for i in range(len(input)):
            if(input[i]):
                index = i + 1
                number = len(input[i])
                print('i is {0} and number is {1}'.format(index,number))
                if(index>= 10 and index <= 16):
                    part = index - 10
                    new_input = [x.split(' ') for x in input[i]]
                    if(number == 1):
                        list.ellipse([dist_to_edge + part * small_box_length, dist_to_top, dist_to_edge + (part+1) * small_box_length, dist_to_top + small_box_length], outline = 'black',fill=new_input[0][0])
                    elif(number == 2):
                        list.ellipse([dist_to_edge + part * small_box_length, dist_to_top, dist_to_edge + (part+1/2) * small_box_length, dist_to_top + 1/2 * small_box_length], outline = 'black',fill=new_input[0][0])
                        list.ellipse([dist_to_edge + (part+1/2) * small_box_length, dist_to_top+ 1/2 * small_box_length, dist_to_edge + (part+1) * small_box_length, dist_to_top + small_box_length], outline = 'black',fill=new_input[1][0])
                    elif(number == 3):
                        list.ellipse([dist_to_edge + part * small_box_length, dist_to_top, dist_to_edge + (part+1/2) * small_box_length, dist_to_top + 1/2 * small_box_length], outline = 'black',fill=new_input[0][0])
                        list.ellipse([dist_to_edge + (part + 1/2) * small_box_length, dist_to_top, dist_to_edge + (part+1) * small_box_length, dist_to_top + 1/2 * small_box_length], outline = 'black',fill=new_input[1][0])
                        list.ellipse([dist_to_edge + (part + 1/2) * small_box_length, dist_to_top+ 1/2 * small_box_length, dist_to_edge + (part+1) * small_box_length, dist_to_top + small_box_length], outline = 'black',fill=new_input[2][0])
                    elif(number == 4):
                        list.ellipse([dist_to_edge + part * small_box_length, dist_to_top, dist_to_edge + (part+1/2) * small_box_length, dist_to_top + 1/2 * small_box_length], outline = 'black',fill=new_input[0][0])
                        list.ellipse([dist_to_edge + (part + 1/2) * small_box_length, dist_to_top, dist_to_edge + (part+1) * small_box_length, dist_to_top + 1/2 * small_box_length], outline = 'black',fill=new_input[1][0])
                        list.ellipse([dist_to_edge + part * small_box_length, dist_to_top+ 1/2 * small_box_length, dist_to_edge + (part+1/2) * small_box_length, dist_to_top + small_box_length], outline = 'black',fill=new_input[2][0])
                        list.ellipse([dist_to_edge + (part + 1/2) * small_box_length, dist_to_top+ 1/2 * small_box_length, dist_to_edge + (part+1) * small_box_length, dist_to_top + small_box_length], outline = 'black',fill=new_input[3][0])
                    else:
                        print ("This is not a valid input")


                #17-23
                elif(index >= 17 and index <=23):
                    part = index - 17
                    new_input = [x.split(' ') for x in input[i]]
                    if(number == 1):
                        list.ellipse([dist_to_edge + 7 * small_box_length, dist_to_top + part * small_box_length, dist_to_edge + 8 * small_box_length, dist_to_top + (part+1) * small_box_length], outline = 'black',fill=new_input[0][0])
                    elif(number == 2):
                        list.ellipse([dist_to_edge + 7 * small_box_length, dist_to_top + part * small_box_length, dist_to_edge + (7+1/2) * small_box_length, dist_to_top + (part +1/2) * small_box_length], outline = 'black',fill=new_input[0][0])
                        list.ellipse([dist_to_edge + (7+1/2) * small_box_length, dist_to_top+ (part+1/2) * small_box_length, dist_to_edge + 8 * small_box_length, dist_to_top + (part + 1) * small_box_length], outline = 'black',fill=new_input[1][0])
                    elif(number == 3):
                        list.ellipse([dist_to_edge + 7 * small_box_length, dist_to_top + part * small_box_length, dist_to_edge + (7+1/2) * small_box_length, dist_to_top + (part +1/2) * small_box_length], outline = 'black',fill=new_input[0][0])
                        list.ellipse([dist_to_edge + (7 + 1/2) * small_box_length, dist_to_top+ (part+1/2) * small_box_length, dist_to_edge + 8 * small_box_length, dist_to_top + (part + 1) * small_box_length], outline = 'black',fill=new_input[1][0])
                        list.ellipse([dist_to_edge + (7 + 1/2) * small_box_length, dist_to_top + part * small_box_length, dist_to_edge + 8 * small_box_length, dist_to_top + (part+ 1/2) * small_box_length], outline = 'black',fill=new_input[2][0])
                    elif(number == 4):
                        list.ellipse([dist_to_edge + 7 * small_box_length, dist_to_top + part * small_box_length, dist_to_edge + (7+1/2) * small_box_length, dist_to_top + (part +1/2) * small_box_length], outline = 'black',fill=new_input[0][0])
                        list.ellipse([dist_to_edge + (7 + 1/2) * small_box_length, dist_to_top+ (part+1/2) * small_box_length, dist_to_edge + 8 * small_box_length, dist_to_top + (part + 1) * small_box_length], outline = 'black',fill = new_input[1][0])
                        list.ellipse([dist_to_edge + (7 + 1/2) * small_box_length, dist_to_top + part * small_box_length, dist_to_edge + 8 * small_box_length, dist_to_top + (part+ 1/2) * small_box_length], outline = 'black',fill=new_input[2][0])
                        list.ellipse([dist_to_edge + 7 * small_box_length, dist_to_top + (part + 1/2) * small_box_length, dist_to_edge + (7 + 1/2) * small_box_length, dist_to_top + (part + 1 ) *small_box_length], outline = 'black',fill=new_input[3][0])
                    else:
                        print ("This is not a valid input")

                #24-30
                elif(index >= 24 and index <= 30):
                    part = index - 24
                    new_input = [x.split(' ') for x in input[i]]
                    if(number == 1):
                        list.ellipse([dist_to_edge + (7-part) * small_box_length, dist_to_top + 7* small_box_length, dist_to_edge + (8-part) * small_box_length, dist_to_top + 8 * small_box_length], outline = 'black',fill=new_input[0][0])
                    elif(number == 2):
                        list.ellipse([dist_to_edge + (7- part) * small_box_length, dist_to_top + 7* small_box_length, dist_to_edge + (7-part+1/2) * small_box_length, dist_to_top + (7+1/2) * small_box_length], outline = 'black',fill=new_input[0][0])
                        list.ellipse([dist_to_edge + (7-part+1/2) * small_box_length, dist_to_top+ (7+1/2) * small_box_length, dist_to_edge + (7-part+1) * small_box_length, dist_to_top + 8 * small_box_length], outline = 'black',fill=new_input[1][0])
                    elif(number == 3):
                        list.ellipse([dist_to_edge + (7- part) * small_box_length, dist_to_top + 7* small_box_length, dist_to_edge + (7-part+1/2) * small_box_length, dist_to_top + (7+1/2) * small_box_length], outline = 'black',fill=new_input[0][0])
                        list.ellipse([dist_to_edge + (7-part+1/2) * small_box_length, dist_to_top+ (7+1/2) * small_box_length, dist_to_edge + (7-part+1) * small_box_length, dist_to_top + 8 * small_box_length], outline = 'black',fill=new_input[1][0])
                        list.ellipse([dist_to_edge + (7-part+1/2) * small_box_length, dist_to_top+ 7 * small_box_length, dist_to_edge + (7-part+1) * small_box_length, dist_to_top + (7+1/2 ) *small_box_length], outline = 'black',fill=new_input[2][0])
                    elif(number == 4):
                        list.ellipse([dist_to_edge + (7- part) * small_box_length, dist_to_top + 7* small_box_length, dist_to_edge + (7-part+1/2) * small_box_length, dist_to_top + (7+1/2) * small_box_length], outline = 'black',fill=new_input[0][0])
                        list.ellipse([dist_to_edge + (7-part+1/2) * small_box_length, dist_to_top+ (7+1/2) * small_box_length, dist_to_edge + (7-part+1) * small_box_length, dist_to_top + 8 * small_box_length], outline = 'black',fill=new_input[1][0])
                        list.ellipse([dist_to_edge + (7-part+1/2) * small_box_length, dist_to_top+ 7 * small_box_length, dist_to_edge + (7-part+1) * small_box_length, dist_to_top + (7+1/2 ) *small_box_length], outline = 'black',fill=new_input[2][0])
                        list.ellipse([dist_to_edge + (7 - part) * small_box_length, dist_to_top+ (7+1/2) * small_box_length, dist_to_edge + (7-part+1/2) * small_box_length, dist_to_top + 8* small_box_length], outline = 'black',fill=new_input[3][0])
                    else:
                        print ("This is not a valid input")


                #31-37
                elif(index >= 31 and index <=37):
                    part = index - 31
                    new_input = [x.split(' ') for x in input[i]]
                    if(number == 1):
                        list.ellipse([dist_to_edge, dist_to_top + (7-part)*small_box_length, dist_to_edge + small_box_length, dist_to_top + (8-part) * small_box_length], outline = 'black',fill=new_input[0][0])
                    elif(number == 2):
                        list.ellipse([dist_to_edge, dist_to_top + (7-part)*small_box_length, dist_to_edge + 1/2 * small_box_length, dist_to_top +(7 - part + 1/2) * small_box_length], outline = 'black',fill=new_input[0][0])
                        list.ellipse([dist_to_edge + (1/2) * small_box_length, dist_to_top+ (7 - part + 1/2) * small_box_length, dist_to_edge + small_box_length, dist_to_top + (8-part) * small_box_length], outline = 'black',fill=new_input[1][0])
                    elif(number == 3):

                        list.ellipse([dist_to_edge, dist_to_top + (7-part)*small_box_length, dist_to_edge + 1/2 * small_box_length, dist_to_top +(7 - part + 1/2) * small_box_length], outline = 'black',fill=new_input[0][0])
                        list.ellipse([dist_to_edge + (1/2) * small_box_length, dist_to_top+ (7 - part + 1/2) * small_box_length, dist_to_edge + small_box_length, dist_to_top + (8-part) * small_box_length], outline = 'black',fill=new_input[1][0])
                        list.ellipse([dist_to_edge + (1/2) * small_box_length, dist_to_top+ (7 - part) * small_box_length, dist_to_edge + small_box_length, dist_to_top + (7- part + 1/2) * small_box_length], outline = 'black',fill=new_input[2][0])
                    elif(number == 4):
                        list.ellipse([dist_to_edge, dist_to_top + (7-part)*small_box_length, dist_to_edge + 1/2 * small_box_length, dist_to_top +(7 - part + 1/2) * small_box_length], outline = 'black',fill=new_input[0][0])
                        list.ellipse([dist_to_edge + (1/2) * small_box_length, dist_to_top+ (7 - part + 1/2) * small_box_length, dist_to_edge + small_box_length, dist_to_top + (8-part) * small_box_length], outline = 'black',fill=new_input[1][0])
                        list.ellipse([dist_to_edge + (1/2) * small_box_length, dist_to_top+ (7 - part) * small_box_length, dist_to_edge + small_box_length, dist_to_top + (7- part + 1/2) * small_box_length], outline = 'black',fill=new_input[2][0])
                        list.ellipse([dist_to_edge, dist_to_top+ (7-part + 1/2) * small_box_length, dist_to_edge + 1/2 * small_box_length, dist_to_top + (8-part) * small_box_length], outline = 'black',fill=new_input[3][0])
                    else:
                        print ("This is not a valid input")


                elif(index >=1 and index <= 3):
                    part = index - 1
                    new_input = [x.split(' ') for x in input[i]]
                    if(number == 1):
                        list.ellipse([dist_to_edge + small_box_length + part * median_box_length, dist_to_top + small_box_length, dist_to_edge + small_box_length + (part +1) * median_box_length, dist_to_top + small_box_length + median_box_length], outline = 'black',fill = new_input[0][0])
                        list.text((dist_to_edge + 2* small_box_length + part * median_box_length, dist_to_top + 2 * small_box_length), text = new_input[0][1],fill = 'black')
                    elif(number == 2):
                        list.ellipse([dist_to_edge + small_box_length + part * median_box_length, dist_to_top + small_box_length, dist_to_edge + small_box_length + (part +1/2) * median_box_length, dist_to_top + small_box_length + 1/2 * median_box_length], outline = 'black',fill= new_input[0][0])
                        list.ellipse([dist_to_edge + small_box_length + (part + 1/2) * median_box_length, dist_to_top + small_box_length + 1/2* median_box_length, dist_to_edge + small_box_length+ (part+1) * median_box_length, dist_to_top + small_box_length + median_box_length], outline = 'black', fill= new_input[1][0])
                    elif(number == 3):
                        list.ellipse([dist_to_edge + small_box_length + part * median_box_length, dist_to_top + small_box_length, dist_to_edge + small_box_length + (part +1/2) * median_box_length, dist_to_top + small_box_length + 1/2 * median_box_length], outline = 'black',fill= new_input[0][0])
                        list.ellipse([dist_to_edge + small_box_length + (part + 1/2) * median_box_length, dist_to_top + small_box_length + 1/2* median_box_length, dist_to_edge + small_box_length+ (part+1) * median_box_length, dist_to_top + small_box_length + median_box_length], outline = 'black',fill=new_input[1][0])
                        list.ellipse([dist_to_edge + small_box_length + (part + 1/2) * median_box_length, dist_to_top + small_box_length, dist_to_edge + small_box_length + (part +1) * median_box_length, dist_to_top + small_box_length + 1/2 * median_box_length], outline = 'black',fill=new_input[2][0])
                    elif(number == 4):
                        list.ellipse([dist_to_edge + small_box_length + part * median_box_length, dist_to_top + small_box_length, dist_to_edge + small_box_length + (part +1/2) * median_box_length, dist_to_top + small_box_length + 1/2 * median_box_length], outline = 'black',fill=new_input[0][0])
                        list.text((dist_to_edge + small_box_length + part * median_box_length, dist_to_top + small_box_length), text = new_input[0][1] )
                        list.ellipse([dist_to_edge + small_box_length + (part + 1/2) * median_box_length, dist_to_top + small_box_length + 1/2* median_box_length, dist_to_edge + small_box_length+ (part+1) * median_box_length, dist_to_top + small_box_length + median_box_length], outline = 'black',fill=new_input[1][0])
                        list.ellipse([dist_to_edge + small_box_length + (part + 1/2) * median_box_length, dist_to_top + small_box_length, dist_to_edge + small_box_length + (part +1) * median_box_length, dist_to_top + small_box_length + 1/2 * median_box_length], outline = 'black',fill=new_input[2][0])
                        list.ellipse([dist_to_edge + small_box_length + part * median_box_length, dist_to_top + small_box_length + 1/2 * median_box_length, dist_to_edge + small_box_length + (part + 1/2 )* median_box_length, dist_to_top + small_box_length + median_box_length], outline = 'black',fill=new_input[3][0])            
                    else:
                        print ("This is not a valid input")

                elif(index >=4 and index <=6):
                    part = index -4
                    new_input = [x.split(' ') for x in input[i]]
                    if(number == 1):
                        list.ellipse([dist_to_edge + small_box_length + (2 - part) * median_box_length, dist_to_top + small_box_length + median_box_length, dist_to_edge + small_box_length + (3-part) * median_box_length, dist_to_top + small_box_length + 2 * median_box_length], outline = 'black', fill = new_input[0][0])
                    elif(number == 2):
                        list.ellipse([dist_to_edge + small_box_length + (2 - part) * median_box_length, dist_to_top + small_box_length + median_box_length, dist_to_edge + small_box_length + (2 - part +1/2) * median_box_length, dist_to_top + small_box_length + 3/2 * median_box_length], outline = 'black', fill = new_input[0][0])
                        list.ellipse([dist_to_edge + small_box_length + (2 - part + 1/2) * median_box_length, dist_to_top + small_box_length + 3/2* median_box_length, dist_to_edge + small_box_length+ (2- part+1) * median_box_length, dist_to_top + small_box_length + 2 * median_box_length], outline = 'black',fill = new_input[1][0])
                    elif(number == 3):
                        list.ellipse([dist_to_edge + small_box_length + (2 - part) * median_box_length, dist_to_top + small_box_length +median_box_length, dist_to_edge + small_box_length + (2 - part +1/2) * median_box_length, dist_to_top + small_box_length + 3/2 * median_box_length], outline = 'black',fill = new_input[0][0])
                        list.ellipse([dist_to_edge + small_box_length + (2 - part + 1/2) * median_box_length, dist_to_top + small_box_length + 3/2* median_box_length, dist_to_edge + small_box_length+ (2 - part+1) * median_box_length, dist_to_top + small_box_length + 2 * median_box_length], outline = 'black',fill = new_input[1][0])
                        list.ellipse([dist_to_edge + small_box_length + (2 - part + 1/2) * median_box_length, dist_to_top + small_box_length + median_box_length, dist_to_edge + small_box_length + (2 - part+1) * median_box_length, dist_to_top + small_box_length + 3/2 * median_box_length], outline = 'black', fill = new_input[2][0])
                    elif(number == 4):
                        list.ellipse([dist_to_edge + small_box_length + (2-part) * median_box_length, dist_to_top + small_box_length + median_box_length, dist_to_edge + small_box_length + (2-part +1/2) * median_box_length, dist_to_top + small_box_length + 3/2 * median_box_length], outline = 'black',fill = new_input[0][0])
                        list.ellipse([dist_to_edge + small_box_length + (2-part + 1/2) * median_box_length, dist_to_top + small_box_length + 3/2* median_box_length, dist_to_edge + small_box_length+ (2-part+1) * median_box_length, dist_to_top + small_box_length + 2*median_box_length], outline = 'black',fill = new_input[1][0])
                        list.ellipse([dist_to_edge + small_box_length + (2-part + 1/2) * median_box_length, dist_to_top + small_box_length + median_box_length, dist_to_edge + small_box_length + (2-part+1) * median_box_length, dist_to_top + small_box_length + 3/2 * median_box_length], outline = 'black', fill = new_input[2][0])
                        list.ellipse([dist_to_edge + small_box_length + (2-part) * median_box_length, dist_to_top + small_box_length + 3/2 * median_box_length, dist_to_edge + small_box_length + (2-part + 1/2 )* median_box_length, dist_to_top + small_box_length + 2* median_box_length], outline = 'black', fill = new_input[3][0])            
                    else:
                        print ("This is not a valid input")

                elif(index >= 7 and index <= 9):
                    part = index - 7
                    new_input = [x.split(' ') for x in input[i]]
                    if(number == 1):
                        list.ellipse([dist_to_edge + small_box_length + part * median_box_length, dist_to_top + small_box_length + 2*median_box_length, dist_to_edge + small_box_length + (part +1) * median_box_length, dist_to_top + small_box_length + 3 * median_box_length], outline = 'black',fill = new_input[0][0])
                    elif(number == 2):
                        list.ellipse([dist_to_edge + small_box_length + part * median_box_length, dist_to_top + small_box_length + 2 * median_box_length, dist_to_edge + small_box_length + (part +1/2) * median_box_length, dist_to_top + small_box_length + 5/2 * median_box_length], outline = 'black', fill = new_input[0][0])
                        list.ellipse([dist_to_edge + small_box_length + (part + 1/2) * median_box_length, dist_to_top + small_box_length + 5/2* median_box_length, dist_to_edge + small_box_length+ (part+1) * median_box_length, dist_to_top + small_box_length + 3 * median_box_length], outline = 'black',fill = new_input[1][0])
                    elif(number == 3):
                        list.ellipse([dist_to_edge + small_box_length + part * median_box_length, dist_to_top + small_box_length + 2* median_box_length, dist_to_edge + small_box_length + (part +1/2) * median_box_length, dist_to_top + small_box_length + 5/2 * median_box_length], outline = 'black',fill = new_input[0][0])
                        list.ellipse([dist_to_edge + small_box_length + (part + 1/2) * median_box_length, dist_to_top + small_box_length + 5/2* median_box_length, dist_to_edge + small_box_length+ (part+1) * median_box_length, dist_to_top + small_box_length + 3 * median_box_length], outline = 'black', fill = new_input[1][0])
                        list.ellipse([dist_to_edge + small_box_length + (part + 1/2) * median_box_length, dist_to_top + small_box_length + 2 * median_box_length, dist_to_edge + small_box_length + (part+1) * median_box_length, dist_to_top + small_box_length + 5/2 * median_box_length], outline = 'black',fill = new_input[2][0])
                    elif(number == 4):
                        list.ellipse([dist_to_edge + small_box_length + part * median_box_length, dist_to_top + small_box_length +2 * median_box_length, dist_to_edge + small_box_length + (part +1/2) * median_box_length, dist_to_top + small_box_length + 5/2 * median_box_length], outline = 'black', fill = new_input[0][0])
                        list.ellipse([dist_to_edge + small_box_length + (part + 1/2) * median_box_length, dist_to_top + small_box_length + 5/2* median_box_length, dist_to_edge + small_box_length+ (part+1) * median_box_length, dist_to_top + small_box_length + 3*median_box_length], outline = 'black',fill = new_input[1][0])
                        list.ellipse([dist_to_edge + small_box_length + (part + 1/2) * median_box_length, dist_to_top + small_box_length + 2 * median_box_length, dist_to_edge + small_box_length + (part+1) * median_box_length, dist_to_top + small_box_length + 5/2 * median_box_length], outline = 'black', fill = new_input[2][0])
                        list.ellipse([dist_to_edge + small_box_length + part * median_box_length, dist_to_top + small_box_length + 5/2 * median_box_length, dist_to_edge + small_box_length + (part + 1/2 )* median_box_length, dist_to_top + small_box_length + 3* median_box_length], outline = 'black', fill = new_input[3][0])            
                    else:
                        print ("This is not a valid input")

        img.save("flashcard_image.png", "PNG")
        return "flashcard_image.png"
