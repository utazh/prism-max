# 实现了一个优先队列结构
def heapify(x):
    """Transform list into a heap, in-place, in O(len(x)) time."""
    n = len(x)
    # Transform bottom-up.  The largest index there's any point to looking at
    # is the largest with a child index in-range, so must have 2*i + 1 < n,
    # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
    # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
    # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
    for i in reversed(range(n//2)):
        _siftup(x, i)

def _siftup(heap, pos,score_pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos].score[score_pos] < heap[rightpos].score[score_pos]:
            childpos = rightpos
        # Move the smaller child up.
        heap[childpos].heap_pos = pos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    newitem.heap_pos = pos
    _siftdown(heap, startpos, pos,score_pos)

def _siftdown(heap, startpos, pos,score_pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem.score[score_pos] < parent.score[score_pos]:
            parent.heap_pos = pos
            heap[pos] = parent
            pos = parentpos
            continue
        break
    newitem.heap_pos = pos
    heap[pos] = newitem

def heappush(heap, item,score_pos):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1,score_pos)

def heappop(heap,score_pos):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0,score_pos)
        return returnitem
    return lastelt

class minInf():
    score = [float('-inf'),float('-inf')]

class maxInf():
    score = [float('inf'),float('inf')]


class Heap: #heap
    def __init__(self,score_pos=0):
        self._queue = []
        self.score_pos=score_pos

    def push(self, item):
        item = item.control
        # heappush 在队列 _queue 上插入第一个元素
        item.heap = self
        heappush(self._queue,item,self.score_pos)

    def pop(self):
        # heappop 在队列 _queue 上删除第一个元素
        item = heappop(self._queue,self.score_pos)
        item.heap = None
        return item.token
    
    def siftup(self,pos):
        _siftup(self._queue,pos,self.score_pos)

    def siftdown(self,pos):
        _siftdown(self._queue, 0, pos,self.score_pos)

    def get_min(self):
        if len(self._queue) == 0:
            return minInf
        return self._queue[0]
    
    def update(self,item):
        _siftup(self._queue, item.control.heap_pos,self.score_pos)
        

    def delete(self,item):
        score = item.score
        item.score = minInf.score
        _siftdown(self._queue, 0, item.heap_pos,self.score_pos)
        heappop(self._queue,self.score_pos)
        item.score = score
        item.heap = None
