# image captcha generation
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# our captchas are comprised of lowercase letters
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


# generate text for captcha
def text_gen(digits_num, char_set=alphabet):
    captcha_text = []
    for i in range(digits_num):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

# generate image for captcha
def image_gen(digits_num, output):
    image = ImageCaptcha()
    captcha_text = text_gen(digits_num=digits_num)
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)    
    image.write(captcha_text, output + captcha_text + '.png')      
    return captcha_text

def data_gen(digits_num,output,total):
# through this function, [total] captchas of [digits_num] named by their digits should be generated in folder [output]
# digits_num is the number of captcha digits
# output is file path in the type of string
# total is the number of captchas generated one time
# the captcha image should be named as [digits].png for training

# possible references: 
# https://www.leiphone.com/news/201706/XaiD0DtlhZuTn8ro.html
# https://github.com/lepture/captcha

    for i in range(total):
        image_gen(digits_num, output)
        

data_gen(4, "images/", 10000)
data_gen(4, "tests/",64)
