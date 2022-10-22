import time
from threading import Thread


class Channel:
    def __init__(self) -> None:
        self.rate = 0


def Channel_rate(channel: Channel):
    while True:
        channel.rate += 2
        # print('time is{}'.format(channel.rate))
        time.sleep(2)


CH = Channel()
t = Thread(target=Channel_rate, args=(CH,), daemon=True)
t.start()
for i in range(20):
    print('now rate = {}'.format(CH.rate))
    time.sleep(1)
