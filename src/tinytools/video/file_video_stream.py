"""FileVideoStream class for efficiently reading from a video file.

Source: https://github.com/PyImageSearch/imutils/blob/master/imutils/video/filevideostream.py
"""

from __future__ import annotations

import time
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


class FileVideoStream:
    """FileVideoStream class for efficiently reading from a video file."""

    def __init__(self, path: str, transform: Callable | None = None, queue_size: int = 128) -> None:
        """Initialize the file video stream class."""
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.transform = transform

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self) -> FileVideoStream:
        """Start a thread to read frames from the file video stream."""
        self.thread.start()
        return self

    def update(self) -> None:
        """Update function for the seperate thread that reads frames from the video stream."""
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True

                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform and frame is not None:
                    frame = self.transform(frame)

                # add the frame to the queue
                self.Q.put(frame)
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self) -> np.ndarray:
        """Return next frame in the queue."""
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self) -> bool:
        """Check if the stream is still running."""
        return self.more() or not self.stopped

    def more(self) -> bool:
        """Check if there are still frames in the queue.

        If stream is not stopped, try to wait a moment.
        """
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self) -> None:
        """Stop the stream."""
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()
