"""
Tkinter UI to compare inference speeds concurrently
"""
import multiprocessing
from threading import Thread
from torchvision.transforms.functional import to_pil_image
from tkinter.ttk import Frame, Label, Button
import argparse
import tkinter
from multiprocessing import Process, Queue
import random
import signal
import torch
import time
import sys
import os
from PIL import ImageTk

import catdog_training

IMAGE_SIZE=256
MODEL_FILE='model_256.pt'


class App(Thread):
    def __init__(self, gui=True):
        Thread.__init__(self)
        self.gui = gui
        self.start_time = 0
        self.am_i_alive = False
        self.die = False
        self.tk_root = None
        self.frame = None
        self.label = {}
        self.button = None

        self.left_frame = None
        self.left_label = {}
        self.left_image = {}
        self.right_label = {}
        self.right_frame = {}
        self.right_image = {}

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
        self.tk_root.protocol('WM_DELETE_WINDOW', self.stop)

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
        """ Update my own UI elements and other periodic tasks """
        elapsed = int(time.time() - self.start_time)
        self.label['text'] = f"Runtime: {elapsed}s"
        if self.gui:
            self.tk_root.after(250, self._tick)
        if self.die:
            self.stop()

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
        # print("make cpu event")
        self.left_queue.put(image)
        self.left_count += 1
        if is_correct:
            self.left_correct += 1
            if self.gui:
                self.tk_root.event_generate("<<cpu_correct>>")
        else:
            if self.gui:
                self.tk_root.event_generate("<<cpu_incorrect>>")

    def nnpa_event(self, image, is_correct: bool):
        """ receive an update from NNPA inference process """
        # print("make nnpa event")
        self.right_queue.put(image)
        self.right_count += 1
        if is_correct:
            self.right_correct += 1
            if self.gui:
                self.tk_root.event_generate("<<nnpa_correct>>")
        else:
            if self.gui:
                self.tk_root.event_generate("<<nnpa_incorrect>>")

    def run(self) -> None:
        if self.gui:
            self.ui_setup()
        self.am_i_alive = True
        self.start_time = time.time()
        self._tick()
        if self.gui:
            print("Running tkinter mainloop...")
            self.tk_root.mainloop()
        else:
            print("Non-GUI main loop:")
            while self.am_i_alive:
                if self.die:
                    self.stop()
                elapsed = int(time.time() - self.start_time)
                sys.stdout.write(f"\rRuntime: {elapsed}s  CPU #: {self.left_count}  NNPA #: {self.right_count}")
                time.sleep(1)
            print()


    def stop(self):
        """ Internal method to kill the app """
        print("App told to stop.")
        self.am_i_alive = False
        if self.gui:
            self.tk_root.destroy()

    def kill(self, *args):
        """ Called externally to make the app stop """
        print("telling app to die")
        self.die = True

    def is_alive(self):
        """ Returns true while the UI is alive. False when it ends. """
        return self.am_i_alive


def run_cpu_process(queue: Queue):
    print(f"CPU PID: {os.getpid()}")
    output_queue = queue
    model, device, test_loader = catdog_training.test_setup(
        batch_size=1, resize=IMAGE_SIZE, num_workers=0, device='cpu', model_path=MODEL_FILE)
    output_queue.get()  # signal to main process that we are ready
    output_queue.get()  # block here until signaled to start
    for input_data, target in test_loader:
        correct, loss = catdog_training.infer_once(model, device, input_data, target)
        output_queue.put((input_data[0], correct))


def run_nnpa_process(queue: Queue):
    print(f"NNPA PID: {os.getpid()}")
    output_queue = queue
    model, device, test_loader = catdog_training.test_setup(
        batch_size=1, resize=IMAGE_SIZE, num_workers=0, device='nnpa', model_path=MODEL_FILE)
    output_queue.get()  # signal to main process that we are ready
    output_queue.get()  # block here until signaled to start
    for input_data, target in test_loader:
        correct, loss = catdog_training.infer_once(model, device, input_data, target)
        output_queue.put((input_data[0], correct))


def get_args():
    parser = argparse.ArgumentParser(description='Compare CPU and NNPA inference speed with tkinter UI')
    parser.add_argument('--cpu-only', action='store_true', default=False,
                        help='Only run the CPU side of the test.')
    parser.add_argument('--nnpa-only', action='store_true', default=False,
                        help='Only run the NNPA side of the test.')
    parser.add_argument('--thread-limit', type=int, default=0,
                        help='Limit pytorch to this many threads. Default=0 means unlimited.')
    parser.add_argument('-n', '--no-gui', action='store_true', default=False,
                        help='Do not launch the GUI.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.thread_limit != 0:
        torch.set_num_threads(args.thread_limit)
    try:
        app = App(gui=not args.no_gui)
        app.start()

        signal.signal(signal.SIGINT, app.kill)

        cpu_output_queue = Queue()
        nnpa_output_queue = Queue()
        cpu_process = None
        nnpa_process = None

        if not args.nnpa_only:
            cpu_output_queue.put(None)
            cpu_process = Process(target=run_cpu_process, args=(cpu_output_queue,))
            cpu_process.start()
        if not args.cpu_only:
            nnpa_output_queue.put(None)
            nnpa_process = Process(target=run_nnpa_process, args=(nnpa_output_queue,))
            nnpa_process.start()

        """
        Inference processes are programmed to setup then pop one item from their output queue.
        Then they wait for an item to be added to it before starting inference.
        Here, we wait until both output queues are empty, then we pop both queues to start them.
        """
        while not (cpu_output_queue.empty() and nnpa_output_queue.empty()):
            time.sleep(0.25)
        print("Both models ready. Starting inference momentarily.")
        time.sleep(3)
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
                message = cpu_output_queue.get_nowait()
                if message is None:
                    print("Warning: None message in CPU output queue")
                else:
                    image_data, is_correct = message
                    app.cpu_event(image_data, is_correct)
                    # print("Sent CPU event")
            if not nnpa_output_queue.empty():
                message = nnpa_output_queue.get_nowait()
                if message is None:
                    print("Warning: None message in NNPA output queue")
                else:
                    image_data, is_correct = message
                    app.nnpa_event(image_data, is_correct)
                    # print("Sent NNPA event")
            time.sleep(0.001)
    except KeyboardInterrupt:
        "caught keyboard interrupt in main loop"
        app.kill()
    finally:
        if cpu_process:
            cpu_process.kill()
        if nnpa_process:
            nnpa_process.kill()


if __name__ == '__main__':
    main()
