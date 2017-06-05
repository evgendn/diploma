import pygame
from random import randint
from pygame.locals import *

from utils import *


class Bird(pygame.sprite.Sprite):
    GRAVITY = 0.6
    STANDART_SPEED = 0.4

    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.velocity = Bird.STANDART_SPEED
        # For stopping bird
        self.moving = True
        self.limit_speed = -8
        self.image = load_png("yellowbird-midflap.png")
        # For reset game
        self.old_x = x
        self.old_y = y

    def update(self):
        if self.moving:
            self.velocity += Bird.GRAVITY
            self.y += self.velocity

    def lift(self):
        self.velocity -= Bird.STANDART_SPEED * 20
        if self.velocity < self.limit_speed:
            self.velocity = self.limit_speed

    def rect(self):
        return Rect(self.x, self.y,
                    self.image.get_width(), self.image.get_height())

    def reset(self):
        self.x = self.old_x
        self.y = self.old_y
        self.velocity = Bird.STANDART_SPEED
        self.moving = True


class Pipe(pygame.sprite.Sprite):
    GAP = 100
    FLUCTUATION = 100
    LOWEST_BORDER = 160
    WIDTH = 52
    HEIGHT = 320

    def __init__(self, x):
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = {}
        self.velocity = 4.5
        # For stopping Pipes
        self.moving = True
        self.top_image = pygame.transform.rotate(
            load_png("pipe-green.png"), 180)
        self.bottom_image = load_png("pipe-green.png")
        self.reposition()
        # Flag for score counting
        self.bird_passed = False
        # For reset game
        self.old_x = self.x
        self.old_y = self.y

    def update(self):
        if self.moving:
            self.x -= self.velocity

            if self.x + Pipe.WIDTH < 0:
                self.reposition()
                # screen width = 288, 1 pipe + 2 gaps = 252
                # 288 - 252 = 36 on  both pipes, 36 / 2 = 16
                # night mindstorm
                self.x = Game.WINDOW_WIDTH + self.GAP + 16

    def reposition(self):
        self.bird_passed = False
        self.y["bottom"] = randint(1, Pipe.FLUCTUATION) + Pipe.GAP \
                           + Pipe.LOWEST_BORDER
        self.y["top"] = self.y["bottom"] - Pipe.GAP \
                        - self.bottom_image.get_height()

    def rect(self):
        top_rect = Rect(self.x, self.y["top"],
                        Pipe.WIDTH, Pipe.HEIGHT)
        bottom_rect = Rect(self.x, self.y["bottom"],
                           Pipe.WIDTH, Pipe.HEIGHT)
        return top_rect, bottom_rect

    def reset(self):
        self.x = self.old_x
        self.y = self.old_y
        self.moving = True
        self.bird_passed = False


class Base:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.image = load_png("base.png")
        self.shift = 0
        self.moving = True

    def update(self):
        if self.moving:
            self.x = -((-self.x + 100) % self.shift)

    def reset(self):
        self.moving = True


class Game:
    WINDOW_WIDTH = 288
    WINDOW_HEIGHT = 512

    def __init__(self):
        # Screen init.
        pygame.init()
        self.screen = pygame.display.set_mode((Game.WINDOW_WIDTH,
                                               Game.WINDOW_HEIGHT))
        pygame.display.set_caption("Flappy Bird clone")
        pygame.mouse.set_visible(False)

        # Game objects
        self.bird = Bird(20, Game.WINDOW_HEIGHT / 2)
        self.pipes = [Pipe(Game.WINDOW_WIDTH),
                      Pipe(Game.WINDOW_WIDTH + Pipe.WIDTH + Pipe.GAP),
                      Pipe(Game.WINDOW_WIDTH + (Pipe.WIDTH + Pipe.GAP) * 2)
                      ]
        self.base = Base(0, self.WINDOW_HEIGHT - 112)
        self.base.shift = self.base.image.get_width() - self.WINDOW_WIDTH
        self.clock = pygame.time.Clock()
        self.score = 0
        self.start = False
        self.background = pygame.transform.scale(load_png("background-black.png"),
                                                 (Game.WINDOW_WIDTH,
                                                  Game.WINDOW_HEIGHT))
        self.score_numbers = [load_png("{}.png".format(i)) for i in range(10)]

    def main_loop(self):
        # Event loop
        while True:
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                elif event.type == KEYDOWN:
                    # Core gameplay
                    if event.key == K_SPACE:
                        self.bird.lift()
                    # Reset game
                    elif event.key == K_r:
                        self.bird.reset()
                        for pipe in self.pipes:
                            pipe.reset()
                        self.base.reset()
                        self.score = 0

            #Check collision
            if self.check_collision():
                self.stop()

            # Check score
            # Check position bird, pipes and increment
            # score if pipe position less then bird.
            # Shitty solution i suppose.
            for pipe in self.pipes:
                if pipe.bird_passed is False:
                    middle_pipe_pos = pipe.x + Pipe.WIDTH / 2
                    middle_bird_pos = self.bird.x + self.bird.image.get_width() / 2
                    if middle_bird_pos > middle_pipe_pos:
                        self.score += 1
                        pipe.bird_passed = True
                        reward = 1

            # Update bird, pipes and base
            self.bird.update()
            for pipe in self.pipes:
                pipe.update()
            self.base.update()

            # Redraw objects
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.bird.image,
                             (self.bird.x, self.bird.y))
            for pipe in self.pipes:
                self.screen.blit(pipe.top_image, (pipe.x, pipe.y["top"]))
                self.screen.blit(pipe.bottom_image, (pipe.x, pipe.y["bottom"]))
            self.screen.blit(self.base.image, (self.base.x, self.base.y))
            self.draw_score()

            # Update display
            pygame.display.flip()

    def next_frame(self, action):
        pygame.event.pump()

        if sum(action) != 1:
            raise ValueError("Multiple input actions!")

        terminal = False
        reward = 0.1

        # action[0] == 1: do nothing
        # action[1] == 1: lift
        if action[1] == 1:
            self.bird.lift()

        # Check collision
        if self.check_collision():
            self.stop()
            terminal = True
            self.__init__()
            reward = -1.0

        # Check score
        # Check position bird, pipes and increment
        # score if pipe position less then bird.
        # Shitty solution i suppose.
        for pipe in self.pipes:
            if pipe.bird_passed is False:
                middle_pipe_pos = pipe.x + Pipe.WIDTH / 2
                middle_bird_pos = self.bird.x + self.bird.image.get_width() / 2
                if middle_bird_pos > middle_pipe_pos:
                    self.score += 1
                    pipe.bird_passed = True
                    reward = 1.0

        # Update bird, pipes and base
        self.bird.update()
        for pipe in self.pipes:
            pipe.update()
        self.base.update()

        # Redraw objects
        # self.screen.fill((0, 0, 0))
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.bird.image,
                         (self.bird.x, self.bird.y))
        for pipe in self.pipes:
            self.screen.blit(pipe.top_image, (pipe.x, pipe.y["top"]))
            self.screen.blit(pipe.bottom_image, (pipe.x, pipe.y["bottom"]))
        self.screen.blit(self.base.image, (self.base.x, self.base.y))
        # self.draw_score()

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        self.clock.tick(30)
        return image_data, reward, terminal

    def check_collision(self):
        # top window border and bird, base and bird
        if self.bird.y + self.bird.image.get_height() > self.base.y:
            self.bird.y = self.base.y - self.bird.image.get_height()
            return True
        elif self.bird.y < 0:
            self.bird.y = 0
            return True

        # Pipes and bird
        for pipe in self.pipes:
            top_rect, bottom_rect = pipe.rect()
            if self.bird.rect().colliderect(top_rect) \
                    or self.bird.rect().colliderect(bottom_rect):
                return True

    def stop(self):
        self.bird.moving = False
        for pipe in self.pipes:
            pipe.moving = False
        self.base.moving = False

    def draw_score(self):
        splitted_score = [int(i) for i in list(str(self.score))]
        # Almost all sprites has width 24 -> half is 12
        width = self.score_numbers[0].get_width()
        offset = (len(splitted_score) * width) / 2
        # Position of score numbers
        x = (Game.WINDOW_WIDTH / 2) - offset
        y = int(Game.WINDOW_HEIGHT / 7)

        for number in splitted_score:
            self.screen.blit(self.score_numbers[number], (x, y))
            x += width

    def get_resolutoin(self):
        return self.WINDOW_WIDTH, self.WINDOW_HEIGHT


def main():
    game = Game()
    game.main_loop()


if __name__ == "__main__":
    main()
