import torch
import adaptedTrain as train
from dataset import main_test


if __name__ == '__main__':
    model = train.model
    model.eval()

    data = main_test()
    dataset = torch.utils.data.DataLoader(data)
    embeddings = {}
    with torch.no_grad():
        for i, (img, label) in enumerate(dataset):
            label = label.numpy()[0]
            print(label)
            assert label == data.targets[i]
            img = img.to(train.device)
            embedding = model(img).cpu().numpy()
            if label not in embeddings.keys():
                embeddings[label] = [embedding]
            else:
                embeddings[label].append(embedding)
        
    print(embeddings)

