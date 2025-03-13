"""
Tkinter UI to compare inference speeds concurrently
"""
import multiprocessing
from threading import Thread
from torchvision.transforms.functional import to_pil_image
from tkinter.ttk import Frame, Label, Button
import tkinter
from multiprocessing import Process, Queue
import random
import time
from PIL import ImageTk

import catdog_training

IMAGE_SIZE=256
MODEL_FILE='model_256.pt'


class App(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.start_time = 0
        self.am_i_alive = False
        self.tk_root = None
        self.frame = None
        self.label = None
        self.button = None

        self.left_frame = None
        self.left_label = None
        self.left_image = None
        self.right_label = None
        self.right_frame = None
        self.right_image = None

        self.left_count = 0
        self.left_correct = 0
        self.left_queue = Queue()
        self.right_count = 0
        self.right_correct = 0
        self.right_queue = Queue()

    def ui_setup(self):
        print("Setting up UI...")
        self.tk_root = tkinter.Tk()
        self.tk_root.bind("<<cpu_correct>>", self._cpu_update)
        self.tk_root.bind("<<cpu_incorrect>>", self._cpu_update)
        self.tk_root.bind("<<nnpa_correct>>", self._nnpa_update)
        self.tk_root.bind("<<nnpa_incorrect>>", self._nnpa_update)

        self.frame = Frame(self.tk_root, padding=10)
        self.frame.grid()
        self.label = Label(self.frame, text="Runtime: ")
        self.label.grid(column=0, row=0)
        self.button = Button(self.frame, text="Quit", command=self.stop)
        self.button.grid(column=1, row=0)

        self.left_frame = Frame(self.frame)
        self.left_frame.grid(column=0, row=1)
        self.left_frame.configure(height=300, width=300)
        self.left_label = Label(self.frame, text="CPU Inference")
        self.left_label.grid(column=0, row=2)
        self.left_image = Label(self.frame)
        self.left_image.grid(column=0, row=1)
        self._cpu_update()

        self.right_frame = Frame(self.frame)
        self.right_frame.grid(column=1, row=1)
        self.right_frame.configure(height=300, width=300)
        self.right_label = Label(self.frame, text="NNPA Inference")
        self.right_label.grid(column=1, row=2)
        self.right_image = Label(self.frame)
        self.right_image.grid(column=1, row=1)
        self._nnpa_update()

    def _tick(self):
        """ Update my own UI elements (just the timer) """
        elapsed = int(time.time() - self.start_time)
        self.label['text'] = f"Runtime: {elapsed}"
        self.tk_root.after(250, self._tick)

    def _cpu_update(self, *args):
        if not self.left_queue.empty():
            tensor = self.left_queue.get_nowait()
            if self.left_count % 10 == 1:
                image = to_pil_image(tensor)
                photo_image = ImageTk.PhotoImage(image)
                self.left_image['image'] = photo_image
                self.left_image.image = photo_image  # need to do this to prevent garbage collection??
        self.left_label['text'] = f"CPU-Inferred Images: {self.left_count}  Correct: {self.left_correct}"

    def _nnpa_update(self, *args):
        if not self.right_queue.empty():
            tensor = self.right_queue.get_nowait()
            if self.right_count % 10 == 1:
                image = to_pil_image(tensor)
                photo_image = ImageTk.PhotoImage(image)
                self.right_image['image'] = photo_image
                self.right_image.image = photo_image  # need to do this to prevent garbage collection??
        self.right_label['text'] = f"NNPA-Inferred Images: {self.right_count}  Correct: {self.right_correct}"

    def cpu_event(self, image, is_correct: bool):
        """ receive an update from CPU inference process """
        self.left_queue.put(image)
        self.left_count += 1
        if is_correct:
            self.left_correct += 1
            self.tk_root.event_generate("<<cpu_correct>>")
        else:
            self.tk_root.event_generate("<<cpu_incorrect>>")

    def nnpa_event(self, image, is_correct: bool):
        """ receive an update from NNPA inference process """
        self.right_queue.put(image)
        self.right_count += 1
        if is_correct:
            self.right_correct += 1
            self.tk_root.event_generate("<<nnpa_correct>>")
        else:
            self.tk_root.event_generate("<<nnpa_incorrect>>")

    def run(self) -> None:
        self.ui_setup()
        self.am_i_alive = True
        print("Running tkinter mainloop...")
        self.start_time = time.time()
        self._tick()
        self.tk_root.mainloop()

    def stop(self):
        self.am_i_alive = False
        self.tk_root.destroy()

    def is_alive(self):
        """ Returns true while the UI is alive. False when it ends. """
        return self.am_i_alive


def run_cpu_process(queue: Queue):
    output_queue = queue
    model, device, test_loader = catdog_training.test_setup(
        batch_size=1, resize=IMAGE_SIZE, num_workers=1, device='cpu', model_path=MODEL_FILE)
    output_queue.get()  # signal to main process that we are ready
    output_queue.get()  # block here until signaled to start
    for input_data, target in test_loader:
        correct, loss = catdog_training.infer_once(model, device, input_data, target)
        output_queue.put((input_data[0], correct))


def run_nnpa_process(queue: Queue):
    output_queue = queue
    model, device, test_loader = catdog_training.test_setup(
        batch_size=1, resize=IMAGE_SIZE, num_workers=1, device='nnpa', model_path=MODEL_FILE)
    output_queue.get()  # signal to main process that we are ready
    output_queue.get()  # block here until signaled to start
    for input_data, target in test_loader:
        correct, loss = catdog_training.infer_once(model, device, input_data, target)
        output_queue.put((input_data[0], correct))


def main():
    app = App()
    app.start()

    cpu_output_queue = Queue()
    nnpa_output_queue = Queue()
    cpu_output_queue.put(None)
    nnpa_output_queue.put(None)

    cpu_process = Process(target=run_cpu_process, args=(cpu_output_queue,))
    cpu_process.start()
    nnpa_process = Process(target=run_nnpa_process, args=(nnpa_output_queue,))
    nnpa_process.start()

    """
    Inference processes are programmed to setup then pop one item from their output queue.
    Then they wait for an item to be added to it before starting inference.
    Here, we wait until both output queues are empty, then we pop both queues to start them.
    """
    while not (cpu_output_queue.empty() and nnpa_output_queue.empty()):
        time.sleep(0.25)
    print("Both models ready. Starting inference.")
    cpu_output_queue.put(None)
    nnpa_output_queue.put(None)

    time.sleep(1)
    print("Starting UI loop")
    while True:
        if not app.is_alive():
            print("App dead. breaking...")
            break
        # app.cpu_event(None, random.choice((True, False)))
        # print("Sent test event")
        if not cpu_output_queue.empty():
            image_data, is_correct = cpu_output_queue.get_nowait()
            app.cpu_event(image_data, is_correct)
            # print("Sent CPU event")
        if not nnpa_output_queue.empty():
            image_data, is_correct = nnpa_output_queue.get_nowait()
            app.nnpa_event(image_data, is_correct)
            # print("Sent NNPA event")
        time.sleep(0.001)
    cpu_process.kill()
    nnpa_process.kill()


if __name__ == '__main__':
    main()
