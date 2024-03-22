import pygame

screen_width, screen_height = 1200, 400
screen_border_width, screen_border_height = 50, 50
fps = 50
speed = 4


def start_screen():

    from math import floor, ceil

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    return screen


def to_screen(*args):
    """
    convert cartesians to screen co-ordinates
    :param args: an XY or a XYWH in normalized cartesian co-ordinates and length on the interval 0..1
    :return: an XY or YYWH in screen co-ordinates
    """
    x_scale, y_scale = screen_width - screen_border_width * 2, screen_height - screen_border_height * 2
    if len(args) == 2:
        x, y = args[0], args[1]

        x = x_scale * x + screen_border_width
        y = y_scale * (1 - y) + screen_border_height
        return x, y

    if len(args) == 4:
        x, y, w, h = args[0], args[1], args[2], args[3]

        x = x_scale * x + screen_border_width
        y = y_scale * (1 - y - h) + screen_border_height
        w = x_scale * w
        h = y_scale * h
        return x, y, w, h

    assert False, "must be a co-ordinate XY or rectangle XYWH"


def draw_rect(screen, xpos, ypos, height, width, color):
    x, y, width, height = to_screen(xpos, ypos, width, height)
    x -= width / 2
    color = pygame.Color(color)
    bar = pygame.Rect(x, y, width, height)
    pygame.draw.rect(screen, color, bar)


def draw_body(screen, body, height, width, color):
    y = 0.0
    draw_rect(screen, body.pos, y, height, width, color)


