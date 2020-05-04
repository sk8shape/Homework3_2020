from torchvision.datasets import ImageFolder
from PIL import Image

Class dataset_PACS(ImageFolder):
    def __init__(self, ):
        data = ImageFolder(root = "Homework3-PACS-master/PACS")
