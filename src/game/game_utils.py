import os
import pygame
import sys


def load_png(name):
    fullname = os.path.join("assets", "sprites", name)
    try:
        image = pygame.image.load(fullname)
        if image.get_alpha() is None:
            image = image.convert()
        else:
            image.convert_alpha()
    except pygame.error:
        print("Cannot load image:", fullname)
        raise SystemExit
    return image


def load_sound(name):
    fullname = os.path.join("assets", "audio", name)
    try:
        sound = pygame.mixer.Sound(fullname)
    except pygame.error:
        print("Cannot load sound:", fullname)
        raise SystemExit
    return sound
