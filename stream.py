import time
import json
import socket
import argparse
import numpy as np
from tqdm import tqdm
import os
from tensorflow.keras.datasets import mnist

parser = argparse.ArgumentParser(description='Streams MNIST data to a Spark Streaming Context')
parser.add_argument('--batch-size', '-b', help='Batch size', required=True, type=int)
parser.add_argument('--endless', '-e', help='Enable endless stream', required=False, type=bool, default=False)
parser.add_argument('--split', '-s', help="training or test split", required=False, type=str, default='train')
parser.add_argument('--sleep', '-t', help="streaming interval", required=False, type=int, default=3)

TCP_IP = "localhost"
TCP_PORT = 6100

class Dataset:
    def __init__(self) -> None:
        self.data = []
        self.labels = []

    def load_mnist_data(self, split: str):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if split == "train":
            self.data = x_train
            self.labels = y_train
        else:
            self.data = x_test
            self.labels = y_test
        self.data = self.data.reshape(-1, 28, 28, 1).astype(np.float32)
        self.data = list(map(np.ndarray.tolist, self.data))
        self.labels = self.labels.tolist()

    def data_generator(self, batch_size: int):
        size_per_batch = (len(self.data) // batch_size) * batch_size
        for ix in range(0, size_per_batch, batch_size):
            image = self.data[ix:ix + batch_size]
            label = self.labels[ix:ix + batch_size]
            yield image, label

    def send_mnist_batch_to_spark(self, tcp_connection, batch_size, split="train"):
        self.load_mnist_data(split)
        total_batches = len(self.data) // batch_size
        pbar = tqdm(total_batches)
        data_received = 0
        for images, labels in self.data_generator(batch_size):
            images = np.array(images).reshape(batch_size, -1).tolist()
            payload = dict()
            for batch_idx in range(batch_size):
                payload[batch_idx] = dict()
                for feature_idx in range(784):  # 28x28 = 784
                    payload[batch_idx][f'feature-{feature_idx}'] = images[batch_idx][feature_idx]
                payload[batch_idx]['label'] = labels[batch_idx]

            payload = (json.dumps(payload) + "\n").encode()
            try:
                tcp_connection.send(payload)
            except BrokenPipeError:
                print("Either batch size is too big or the connection was closed")
            except Exception as error_message:
                print(f"Exception thrown but was handled: {error_message}")

            data_received += 1
            pbar.update(n=1)
            pbar.set_description(f"it: {data_received} | received: {batch_size} images")
            time.sleep(sleep_time)

    def connect_tcp(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"Waiting for connection on port {TCP_PORT}...")
        connection, address = s.accept()
        print(f"Connected to {address}")
        return connection, address

    def stream_mnist_dataset(self, tcp_connection, batch_size, split):
        self.send_mnist_batch_to_spark(tcp_connection, batch_size, split)

if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = args.batch_size
    endless = args.endless
    sleep_time = args.sleep
    train_test_split = args.split
    dataset = Dataset()
    tcp_connection, _ = dataset.connect_tcp()
    
    if endless:
        while True:
            dataset.stream_mnist_dataset(tcp_connection, batch_size, train_test_split)
    else:
        dataset.stream_mnist_dataset(tcp_connection, batch_size, train_test_split)
    
    tcp_connection.close()