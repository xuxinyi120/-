class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'mine':
            # folder that contains class labels
            root_dir = r'/mnt/nvme1n1/xxy/LRCN/rawdata'

            # Save preprocess data into output_dir
            output_dir = r'/mnt/nvme1n1/xxy/LRCN/preprocess'

            return root_dir, output_dir
        
        #elif database == 'hmdb51':
            # folder that contains class labels
        #    root_dir = '/Path/to/hmdb-51'

        #    output_dir = '/path/to/VAR/hmdb51'

        #    return root_dir, output_dir
        
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './C3Dmodel/C3D-ucf_epoch-82.pth.tar'