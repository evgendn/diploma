import os
import pygame


def load_png(name):
    fullname = os.path.join("assets", "sprites", name)
    try:
        image = pygame.image.load(fullname)
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
