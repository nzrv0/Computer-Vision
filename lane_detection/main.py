from model import LaneNet
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from dataset import LaneDataset
from torch.optim import SGD
from helpers import get_path
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(learning_rate=0.001):
    model = LaneNet()
    loss_fn = CrossEntropyLoss()
    optim = SGD(model.parameters(), learning_rate)
    return model, loss_fn, optim


def load_dataset():
    training = "training"
    cords = "cords"

    traning_data = DataLoader(
        LaneDataset(training, cords, device), batch_size=32, shuffle=True
    )

    return traning_data


def train_batch(x, y, model, loss, optim):
    model.train()
    pred = model(x)
    loss_fn = loss(y, pred)
    loss_fn.backward()
    optim.step()
    optim.zero_grad()

    return loss_fn.item()


if __name__ == "__main__":
    model_path = get_path("weights")
    model_name = model_path / "first_model.pth"
    traning_data = load_dataset()
    model, loss_fn, optim = load_model()
    total_loss = []
    loss_per_data = []
    for index, batch in enumerate(iter(traning_data)):
        for item in tqdm(batch):
            x, y = item
            print(item[x].shape)
            print(item[y])
            loss = train_batch(item[x], item[y], model, loss_fn, optim)
            loss_per_data.append(loss)
        total_loss.append(sum(loss_per_data))
        torch.save(model.state_dict(), model_name)
