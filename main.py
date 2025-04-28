from trainer import SparkConfig, Trainer
from models import CNN
from transforms import Transforms, Normalize

# Khởi tạo các phép biến đổi
transforms = Transforms([
    Normalize(mean=0.1307, std=0.3081)  # Mean và std của MNIST
])

if __name__ == "__main__":
    spark_config = SparkConfig()
    cnn = CNN()
    trainer = Trainer(cnn, "train", spark_config, transforms)
    trainer.train()