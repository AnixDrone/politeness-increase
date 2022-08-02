from torch.utils.data import Dataset



class ProcessedSentences(Dataset):
    
    def __init__(self,input_data,output_data) -> None:
        super().__init__()
        
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self,idx):
        return self.input_data[idx],self.output_data[idx]