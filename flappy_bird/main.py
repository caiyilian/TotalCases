import flappy_bird
from multiprocessing import Process
import os

if __name__ == '__main__':
    os.environ['SDL_VIDEO_WINDOW_POS'] = "350,50"
    p = Process(target=flappy_bird.man)  # 设置子进程执行的函数，实例化一个对象绑定名称。
    p.start()  # 启动子进程
    os.environ['SDL_VIDEO_WINDOW_POS'] = "50,50"
    flappy_bird.robot()
