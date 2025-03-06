"""
Tkinter UI to compare inference speeds concurrently
"""
import multiprocessing
from threading import Thread
from tkinter.ttk import Frame, Label, Button
import tkinter
from queue import Queue
import random
import time


class App(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.am_i_alive = False
        self.tk_root = None
        self.frame = None
        self.left_frame = None
        self.left_label = None
        self.right_label = None
        self.right_frame = None
        self.label = None
        self.button = None
        self.left_image = None
        self.left_count = 0
        self.left_correct = 0
        self.left_queue = Queue()

    def ui_setup(self):
        print("Setting up UI...")
        self.tk_root = tkinter.Tk()
        self.tk_root.bind("<<cpu_correct>>", self._cpu_update)
        self.tk_root.bind("<<cpu_incorrect>>", self._cpu_update)

        self.frame = Frame(self.tk_root, padding=10)
        self.frame.grid()
        self.label = Label(self.frame, text="Hello World!")
        self.label.grid(column=0, row=0)
        self.button = Button(self.frame, text="Quit", command=self.tk_root.destroy)
        self.button.grid(column=1, row=0)

        self.left_frame = Frame(self.frame)
        self.left_frame.grid(column=0, row=1)
        self.left_frame.configure(height=600, width=600)
        self.left_label = Label(self.frame, text="CPU Inference")
        self.left_label.grid(column=0, row=2)
        self._cpu_update()

        self.right_frame = Frame(self.frame)
        self.right_frame.grid(column=1, row=1)
        self.right_frame.configure(height=600, width=600)
        self.right_label = Label(self.frame, text="NNPA Inference")
        self.right_label.grid(column=1, row=2)

    def _cpu_update(self, *args):
        """ receive an update from CPU inference process """
        if not self.left_queue.empty():
            self.left_image = self.left_queue.get()
        self.left_label['text'] = f"CPU-Inferred Images: {self.left_count}  Correct: {self.left_correct}"

    def cpu_event(self, image, is_correct: bool):
        self.left_queue.put(image)
        self.left_count += 1
        if is_correct:
            self.left_correct += 1
            self.tk_root.event_generate("<<cpu_correct>>")
        else:
            self.tk_root.event_generate("<<cpu_incorrect>>")

    def run(self) -> None:
        self.ui_setup()
        self.am_i_alive = True
        print("Running tkinter mainloop...")
        self.tk_root.mainloop()
        self.am_i_alive = False

    def is_alive(self):
        """ Returns true while the UI is alive. False when it ends. """
        return self.am_i_alive


def main():
    app = App()
    app.start()
    time.sleep(1)
    while True:
        if not app.is_alive():
            print("App dead. breaking...")
            break
        time.sleep(1)
        app.cpu_event(None, random.choice((True, False)))
        print("Sent test event")



if __name__ == '__main__':
    main()