import uuid


def next_id(length=8):
    while True:
        uid = str(uuid.uuid4())[:length]  # Generate UUID and truncate it
        yield uid


class TimerQueue:
    def __init__(self):
        self.queue = []

    def push(self, timer):
        self.queue.append(timer)
        self.queue.sort(key=lambda x: x.t)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from an empty priority queue")
        return self.queue.pop(0)  # Pop and return the item with the highest priority

    def peek(self):
        if self.is_empty():
            return None
        return self.queue[0]  # Return the item with the highest priority

    def is_empty(self):
        return len(self.queue) == 0

    def cancel(self, id):
        for i, timer in enumerate(self.queue):
            if timer.id == id:
                del self.queue[i]

    def __repr__(self):
        return str(self.queue)


class Timer:
    def __init__(self, t, timer_queue):
        """
        :param t: the absolute time at which the timer will expire

        To use just create the object, it will automatically be added to the timer queue
        """
        self.t = t
        self.id = next(next_id())
        self.timer_queue = timer_queue
        timer_queue.push(self)

    def on_expire(self):
        """
        called if timer expires
        """
        pass

    def cancel(self):
        """
        cancels this timer
        """
        self.timer_queue.cancel(self.id)

    def __repr__(self):
        return f'{self.__class__.__name__}(t={self.t}, id={self.id})'
