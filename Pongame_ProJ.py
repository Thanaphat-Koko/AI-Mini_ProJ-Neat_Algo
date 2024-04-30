import pygame
from pong import Game
import neat
import os
import pickle

font = pygame.font.SysFont("comicsans", 50)

#! class สำหรับเรียกใช้เกม Pingpong
class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball
        self.gameOver = False
        self.winMSG = ""

    #? Function สำหรับทดสอบ AI โดยจะแข่งกับผู้เล่น
    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
                if self.gameOver:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.game.left_score = 0
                            self.game.right_hits = 0
                            time = 0
                            self.gameOver = False
            
            if not self.gameOver:
                time = pygame.time.get_ticks() // 1000

                keys = pygame.key.get_pressed()
                if keys[pygame.K_w]:
                    self.game.move_paddle(left=True, up=True)
                if keys[pygame.K_s]:
                    self.game.move_paddle(left=True, up=False)

                output = net.activate(
                    (self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
                decision = output.index(max(output))

                if decision == 0:
                    pass
                elif decision == 1:
                    self.game.move_paddle(left=False, up=True)
                else:
                    self.game.move_paddle(left=False, up=False)

                game_info = self.game.loop()
                self.game.draw(True, False)

                left_score = self.game.left_score
                right_score = self.game.right_score

                if left_score >= 5:
                    self.winMsg = "Human Wins!"
                    self.gameOver = True
                elif right_score >= 5:
                    self.winMsg = "AI Wins!"
                    self.gameOver = True

                if time == 30:
                    if left_score > right_score:
                        self.winMsg = "Human Wins!"
                        self.gameOver = True
                    elif right_score > left_score:
                        self.winMsg = "AI Wins!"
                        self.gameOver = True
                    else:
                        self.winMsg = "DRAW!"
                        self.gameOver = True

            self.game.window.blit(font.render("Time: "+str(time), True, self.game.WHITE), (self.game.window_width//2-80, 0))
            if self.gameOver:
                self.game.window.blit(font.render(self.winMsg, True, self.game.RED), (self.game.window_width//2-80, self.game.window_height//2-20))
                    
            pygame.display.update()

        pygame.quit()

    #? Function สำหรับการ train AI โดยจะให้ AI 2 ตัวแข่งกันหาค่า fitness
    def train_ai(self, genome1, genome2, config):
        # AI ตัวที่ 1
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        # AI ตัวที่ 2
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            
            # AI ตัวที่ 1 ตัดสินใจ
            output1 = net1.activate(
                (self.left_paddle.y, self.ball.y, abs(self.left_paddle.x - self.ball.x)))
            decision1 = output1.index(max(output1))

            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            # AI ตัวที่ 2 ตัดสินใจ
            output2 = net2.activate(
                (self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision2 = output2.index(max(output2))

            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()

            self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()

            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                self.calculate_fitness(genome1, genome2, game_info)
                break
    #? Function สำหรับคำนวณค่า fitness
    def calculate_fitness(self, genome1, genome2, game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits

#! function สำหรับการวิวัฒนาการของ AI
def eval_genomes(genomes, config):
    width, height = 960, 540
    window = pygame.display.set_mode((width, height))

    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = PongGame(window, width, height)
            game.train_ai(genome1, genome2, config)

#! function สำหรับการ train AI
def run_neat(config):
    # train ใหม่ตั้งแต่เริ่มต้น
    #p = neat.Population(config)
    
    #train ต่อจาก checkpoint ล่าุสด
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-39')

    #แสดงรายละเอียดการ train
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    #กำหนดจำนวน generation ที่จะ train
    winner = p.run(eval_genomes, 40)
    #และนำ AI หรือ genome ที่เก่งที่สุดไปเก็บลงในไฟล์ best.pickle
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
   
#! function สำหรับการเล่นเพื่อทดสอบ AI
def play_with_ai(config):
    width, height = 960, 540
    window = pygame.display.set_mode((width, height))

    #เลือก AI ที่เก่งที่สุดในไฟล์ best.pickle
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)

    pygame.display.set_caption("Pong")
    game = PongGame(window, width, height)
    game.test_ai(winner, config)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    #run_neat(config)
    play_with_ai(config)
