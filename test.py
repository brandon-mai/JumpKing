from collections import deque

queue = deque(maxlen=10)
for i in range(20):
    queue.append(i)
    print(list(queue))

for _ in range(5):
    queue.pop()
    print(list(queue))