import pygame as pg
import sys
import network as nn
import numpy as np
import time
import random

pg.init()

HEIGHT = 400
WIDTH = 400
BLACK = 0
WHITE = 255

p_size = 10

weights_and_biases = np.load("./NN_weights_and_biases.npz")
w1 = weights_and_biases['arr_0']
b1 = weights_and_biases['arr_1']
w2 = weights_and_biases['arr_2']
b2 = weights_and_biases['arr_3']
w3 = weights_and_biases['arr_4']
b3 = weights_and_biases['arr_5']

displaysurface = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Number Guesser w/ a NN")
displaysurface.fill((255, 100, 25))
pg.display.update()


def draw(pixels):
    for i in range(len(pixels)):
       pg.draw.rect(displaysurface,
                    (pixels[i][0], pixels[i][0], pixels[i][0]),
                    ((WIDTH - p_size*28)/2 + (i % 28)*p_size, (HEIGHT - p_size*28)/2 + int(i / 28)*p_size, p_size, p_size))
    pg.display.update()


def reset_pixels(pixels):
    if len(pixels) == 0:
        for i in range(0, 784):
            pixels.append([BLACK])
    else:
        for i in range(0, 784):
            pixels[i][0] = BLACK
    draw(pixels)


def update_pixels(pixels, x, y):
    x_rel = x - (WIDTH - p_size*28)/2
    y_rel = y - (HEIGHT - p_size*28)/2

    if (0 < x_rel < p_size*28) and (0 < y_rel < p_size*28):
        i = int(28 * int(y_rel / p_size) + int(x_rel / p_size))
        pixels[i][0] = WHITE
        if 0 < i < len(pixels) - 1:
            if pixels[i - 1][0] == BLACK:
                pixels[i - 1][0] = WHITE - random.random() * 200
            if pixels[i + 1][0] == BLACK:
                pixels[i + 1][0] = WHITE - random.random() * 200
        draw(pixels)


def send_data():
    return pixels


pixels = []
reset_pixels(pixels)

pg.display.update()

data = np.array(pixels)/255.0

run = True
while run:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()

    keys = pg.key.get_pressed()

    if keys[pg.K_SPACE]:
        reset_pixels(pixels)

    if keys[pg.K_p]:
        data = np.array(pixels)/255.0
        z1, a1, z2, a2, z3, a3 = nn.forward_prop(w1, b1, w2, b2, w3, b3, data)
        prediction = nn.get_predictions(a3)
        print(a3*100, prediction)
        time.sleep(1)

    if pg.mouse.get_pressed()[0]:
        update_pixels(pixels, pg.mouse.get_pos()[0], pg.mouse.get_pos()[1])


