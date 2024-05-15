
import dask.dataframe as dd

class ProcessDataframe():
    def __init__(self, data_file):
        self.dataframe = self.read_data_dask(data_file)
        self.data_file = data_file
        
        
    def read_data_dask(self, data_file, blocksize=2**28):
        return dd.read_json(data_file, lines=True, blocksize=blocksize)
            
    def split_dataframe(self, n_splits):
        # Get the number of partitions
        n_partitions = self.dataframe.npartitions
        
        # Compute how many partitions per split
        partitions_per_split = n_partitions // n_splits
        
        # Split the dataframe into n_splits
        dfs = []
        for i in range(n_splits):
            start = i * partitions_per_split
            end = (i + 1) * partitions_per_split if i < n_splits - 1 else n_partitions
            dfs.append(self.dataframe.partitions[start:end])
        return dfs

    def get_shape(self, df):
        return df.shape[0].compute(), df.shape[1]
    
    def get_data(self):
        return self.dataframe.compute()
    
    def save_to_disk(self, df, write_file):
        df.to_parquet(write_file)
    
    def load_from_disk(self, read_file):
        return dd.read_parquet(read_file)
    
    
def main():
    data_file = './data_mlm/process_folder/list_content_word_v2/NOUN.json'
    process = ProcessDataframe(data_file)
    
    # Split the dataframe into 3 and save to disk
    dfs = process.split_dataframe(3)
    
    for i, df in enumerate(dfs):
        process.save_to_disk(df, f'{i}.parquet')
        
if __name__ == '__main__':
    main()