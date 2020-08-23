import time


# Timer counting seconds since start
# can be reused multiple times and get average runtime by repeatedly calling start() + stop()
# can be paused via calling pause() between start() and stop()
class Timer:
    def __init__(self, name=""):
        self.name = "Timer" + name if name else str(time.time())
        self.start_time = None
        self.paused = False
        self.stopped = False
        self.paused_time = None
        self.total_paused = 0
        self.delta_avg = 0
        self.use_count = 0
        self.delta = 0

    def reset(self):
        self.__init__()

    def restart(self):
        self.paused = False
        self.stopped = False
        self.paused_time = None
        self.total_paused = 0

    def start(self):
        self.restart()
        self.start_time = time.time()

    def stop_start(self):
        self.stop()
        self.start()

    def pause(self):
        if not self.paused:
            self.paused = True
            self.paused_time = time.time()

    def unpause(self):
        if self.paused:
            self.total_paused += time.time() - self.paused_time
            self.paused_time = None
        self.paused = False

    def stop(self):
        if not self.stopped:
            self.stopped = True
            self.delta = time.time() - self.start_time - self.total_paused
            self.delta_avg += 1.0 / (self.use_count + 1) * (self.delta - self.delta_avg)
            self.use_count += 1
        return self.delta
