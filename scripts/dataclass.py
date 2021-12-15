# toch Dataset Class
import torch

class IMDbReviews(torch.utils.data.Dataset):
  ''' Dataset for our IMDb revies in spanish 
  '''

  def __init__(self, encodings, labels):
      self.encodings = encodings
      self.labels = labels

  def __getitem__(self, idx):
      item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
      item['labels'] = torch.tensor(self.labels[idx])
      return item

  def __len__(self):
      return len(self.labels)